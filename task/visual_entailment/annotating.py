import os
import re
import copy
import time
import requests
import base64
from openai import OpenAI
from PIL import Image
import requests
import pickle
from io import BytesIO
from tqdm.auto import tqdm
# Custom Modules
from utils.utils import check_path

client = OpenAI(api_key=os.environ['OPENAI_API_KEY_ACL'])
domain_map = {
    0: 'realistic photo', # default image
    1: 'cartoon drawing', # cartoon drawing
    2: 'pencil drawing', # pencil drawing
    3: 'oil painting', # oil painting
}

def annotating_ve(args):
    assert args.task_dataset in ['snli_ve_sports_ori'], "Only support original datasets as a source to annotate."
    assert args.test_dataset in ['snli_ve_sports_cartoon', 'snli_ve_sports_pencil', 'snli_ve_sports_oil'], "Only support cartoon, pencil, and oil painting as target domains."

    if args.test_dataset == 'snli_ve_sports_cartoon':
        args.target_domain_id = 1
    elif args.test_dataset == 'snli_ve_sports_pencil':
        args.target_domain_id = 2
    elif args.test_dataset == 'snli_ve_sports_oil':
        args.target_domain_id = 3

    # load train, valid, test data
    with open(os.path.join(args.preprocess_path, args.task_dataset, f'train_data.pkl'), 'rb') as f:
        train_ori_data = pickle.load(f)
    with open(os.path.join(args.preprocess_path, args.task_dataset, f'valid_data.pkl'), 'rb') as f:
        valid_ori_data = pickle.load(f)
    with open(os.path.join(args.preprocess_path, args.task_dataset, f'test_data.pkl'), 'rb') as f:
        test_ori_data = pickle.load(f)
    check_path(os.path.join(args.preprocess_path, args.test_dataset))

    data_dict = {
        'train': {
            'image': train_ori_data['image'],
            'image_id': train_ori_data['image_id'],
            'premise': train_ori_data['premise'],
            'hypothesis': train_ori_data['hypothesis'],
            'label': train_ori_data['label'],
            'domain_id': train_ori_data['domain_id'],
        },
        'valid': {
            'image': valid_ori_data['image'],
            'image_id': valid_ori_data['image_id'],
            'premise': valid_ori_data['premise'],
            'hypothesis': valid_ori_data['hypothesis'],
            'label': valid_ori_data['label'],
            'domain_id': valid_ori_data['domain_id'],
        },
        'test': {
            'image': test_ori_data['image'],
            'image_id': test_ori_data['image_id'],
            'premise': test_ori_data['premise'],
            'hypothesis': test_ori_data['hypothesis'],
            'label': test_ori_data['label'],
            'domain_id': test_ori_data['domain_id'],
        }
    }

    # Every element should have same length
    assert len(data_dict['train']['image']) == len(data_dict['train']['image_id']) == len(data_dict['train']['premise']) == len(data_dict['train']['hypothesis']) == len(data_dict['train']['label']) == len(data_dict['train']['domain_id'])
    assert len(data_dict['valid']['image']) == len(data_dict['valid']['image_id']) == len(data_dict['valid']['premise']) == len(data_dict['valid']['hypothesis']) == len(data_dict['valid']['label']) == len(data_dict['valid']['domain_id'])
    assert len(data_dict['test']['image']) == len(data_dict['test']['image_id']) == len(data_dict['test']['premise']) == len(data_dict['test']['hypothesis']) == len(data_dict['test']['label']) == len(data_dict['test']['domain_id'])
    print(len(list(set(data_dict['train']['image_id']))), len(list(set(data_dict['valid']['image_id']))), len(list(set(data_dict['test']['image_id']))))

    # for split in ['valid', 'test', 'train']:
    for split in ['valid', 'test', 'train']:
        new_data = {
            'image': [],
            'image_id': [],
            'premise': [],
            'hypothesis': [],
            'label': [],
            'domain_id': [],
            'original_prompt': [],
            'style_prompt': [],
        }

        tqdm_bar = tqdm(range(len(data_dict[split]['image'])), desc=f'Annotating {split} data')
        index = 0
        while index < len(data_dict[split]['image']):
            image = convert_to_base64(data_dict[split]['image'][index].resize((args.image_resize_size, args.image_resize_size)))
            prompt = build_default_prompt(args)

            # VE Annotation Step 1. Generate prompt for the given image
            original_prompt = generate_original_prompt(args, image, prompt)
            if original_prompt is not None:
                original_prompt, prompt = original_prompt # unpack the tuple
            else:
                tqdm.write(f'[FAILURE - generate_original_prompt] Error in generating original prompt. Skipping the index {index}...')
                index += 1
                continue

            # VE Annotation Step 2. Modify the generated prompt to change the style of the image
            style_prompt = generate_style_prompt(args, prompt)
            if style_prompt is not None:
                style_prompt, prompt = style_prompt # unpack the tuple
            else:
                tqdm.write(f'[FAILURE - generate_style_prompt] Error in generating style prompt. Skipping the index {index}...')
                index += 1
                continue

            # VE Annotation Step 3. Generate the image based on the modified prompt using DALL-E3 model
            # VE Annotation Step 4. Verify if the generated image is a {domain_map[args.target_domain_id]} style image of the original image
            generation_error_count = 0
            while True:
                generated_image = generate_image(args, style_prompt)
                if generated_image is None:
                    tqdm.write(f'[Error - generate_image] Error in generating image. Re-trying...')
                    break

                # VE Annotation Step 4. Verify if the generated image is a {domain_map[args.target_domain_id]} style image of the original image
                verification = verify_image(args, prompt, generated_image)
                if verification is None:
                    generation_error_count += 1
                    if generation_error_count >= 3:
                        tqdm.write(f'[Error - verify_image] Error in verifying image. Forcing to use current image...')
                        break
                    else:
                        tqdm.write(f"[Message - verify_image] Generated image is not verified. Re-trying...")
                        continue
                else:
                    prompt = verification
                    break
                break
            if generated_image is None:
                index += 1
                continue # move to the next image

            # VE Annotation Step 5. Check if the premise-hypothesis pair is valid for the generated image
            i = index
            # image_universal_prompt = copy.deepcopy(prompt) # keep the universal prompt for the same image_id to avoid explosion of prompts
            prompt = build_default_prompt(args) # reset the prompt for Step 5 to reduce the cost
            # perform the annotation for the same image_id
            while (data_dict[split]['image_id'][i] == data_dict[split]['image_id'][index]) and i < len(data_dict[split]['image']):
                if data_dict[split]['label'][i] == 0:
                    label_text = 'True'
                elif data_dict[split]['label'][i] == 1:
                    label_text = 'Undetermined'
                elif data_dict[split]['label'][i] == 2:
                    label_text = 'False'

                ve_verification = verify_hypothesis(args, prompt, generated_image, data_dict[split]['hypothesis'][i], label_text, domain_map[args.target_domain_id])
                if ve_verification is None:
                    tqdm.write(f'[FAILURE - verify_hypothesis] Error in verifying hypothesis. Skipping the data index {i}...')
                    i += 1
                    continue
                elif ve_verification[0] == True:
                    ve_verification, _ = ve_verification # unpack the tuple
                    paraphrased_hypothesis = generate_paraphrased_hypothesis(args, prompt, generated_image, data_dict[split]['hypothesis'][i], domain_map[args.target_domain_id])
                    if paraphrased_hypothesis is None:
                        tqdm.write(f'[FAILURE - generate_paraphrased_hypothesis] Error in generating paraphrased hypothesis. Skipping the data index {i}...')
                        i += 1
                        continue
                    else:
                        paraphrased_hypothesis, _ = paraphrased_hypothesis

                    new_data['image'].append(generated_image.convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                    new_data['image_id'].append(data_dict[split]['image_id'][i])
                    new_data['premise'].append(data_dict[split]['premise'][i])
                    new_data['hypothesis'].append(paraphrased_hypothesis)
                    new_data['label'].append(data_dict[split]['label'][i])
                    new_data['domain_id'].append(args.target_domain_id)
                    new_data['original_prompt'].append(original_prompt)
                    new_data['style_prompt'].append(style_prompt)
                else:
                    ve_verification = False

                    # VE Annotation Step 5b1. If the hypothesis/label pair is not valid, directly annotate the hypothesis/label pair
                    ve_correction = annotate_entailment(args, prompt, generated_image, data_dict[split]['hypothesis'][i])
                    if ve_correction is None:
                        tqdm.write(f'[FAILURE - annotate_entailment] Error in annotating entailment. Skipping the data index {i}...')
                        i += 1
                        continue
                    else:
                        ve_correction, _ = ve_correction

                    paraphrased_hypothesis = generate_paraphrased_hypothesis(args, prompt, generated_image, data_dict[split]['hypothesis'][i], domain_map[args.target_domain_id])
                    if paraphrased_hypothesis is None:
                        tqdm.write(f'[FAILURE - generate_paraphrased_hypothesis] Error in generating paraphrased hypothesis. Skipping the data index {i}...')
                        i += 1
                        continue
                    else:
                        paraphrased_hypothesis, _ = paraphrased_hypothesis

                    new_data['image'].append(generated_image.convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                    new_data['image_id'].append(data_dict[split]['image_id'][i])
                    new_data['premise'].append(data_dict[split]['premise'][i])
                    new_data['hypothesis'].append(paraphrased_hypothesis)
                    new_data['label'].append(ve_correction)
                    new_data['domain_id'].append(args.target_domain_id)
                    new_data['original_prompt'].append(original_prompt)
                    new_data['style_prompt'].append(style_prompt)

                i += 1
                if i >= len(data_dict[split]['image']):
                    break
                tqdm_bar.update(1)
                continue # move to the next hypothesis/label pair

            index = i # move to the next image_id
            prompt = None # reset the prompt
            if len(list(set(new_data['image_id']))) % 10 == 0 or index >= len(data_dict[split]['image']):
                with open(os.path.join(args.preprocess_path, args.test_dataset, f'{split}_temp.pkl'), 'wb') as f:
                    pickle.dump(new_data, f)
                    tqdm.write(f'[MESSAGE - Saving Temp Data] Saved {split} temp data')

            if index >= len(data_dict[split]['image']):
                break

        # Save the annotated data
        with open(os.path.join(args.preprocess_path, args.test_dataset, f'{split}_data.pkl'), 'wb') as f:
            pickle.dump(new_data, f)
        tqdm.write(f'[MESSAGE - Saving Data] Saved {split} data')

def encode_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

def convert_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_original_prompt(args, image, prompt):
    error_counter = 0

    prompt.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Please generate a detailed prompt for DALL-E3 model to replicate the given image."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                    "detail": "low"
                }
            }
        ]
    })

    while True:
        try:
            response = client.chat.completions.create(
                model=args.gpt_model_version,
                messages=prompt
            )
            break
        except Exception as e:
            tqdm.write(f"[Error - generate_original_prompt] {e}")
            time.sleep(0.1) # wait for 0.1 seconds
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue # re-try to generate the prompt

    prompt.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response.choices[0].message.content
            }
        ]
    })

    original_prompt = response.choices[0].message.content
    tqdm.write(f"[Result - generate_original_prompt] {original_prompt}")
    return original_prompt, prompt

def generate_style_prompt(args, prompt):
    error_counter = 0

    prompt.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please modify the generated prompt to change the style of the image to a {domain_map[args.target_domain_id]} style."
            }
        ]
    })

    while True:
        try:
            response = client.chat.completions.create(
                model=args.gpt_model_version,
                messages=prompt
            )
            break
        except Exception as e:
            tqdm.write(f"[Error - generate_style_prompt] {e}")
            time.sleep(0.1)
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue

    prompt.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response.choices[0].message.content
            }
        ]
    })

    style_prompt = response.choices[0].message.content
    tqdm.write(f"[Result - generate_style_prompt] {style_prompt}")
    return style_prompt, prompt

def generate_image(args, image_prompt):
    error_counter = 0

    while True:
        try:
            response = client.images.generate(
                model=args.dalle_model_version,
                prompt=image_prompt,
                size="1024x1024", quality="standard", n=1
            )
            generated_image = get_image_from_url(response.data[0].url)
            break
        except Exception as e:
            time.sleep(0.1)
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue

    return generated_image

def verify_image(args, prompt, generated_image):
    error_counter = 0
    local_prompt = copy.deepcopy(prompt)

    base64_generated_image = convert_to_base64(generated_image.resize((args.image_resize_size, args.image_resize_size)))
    local_prompt.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please verify if the image below is a {domain_map[args.target_domain_id]} style image of the original image. The generated image should not exactly match the original image but should capture the essence of the original image. Start the response with 'Yes' or 'No'."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_generated_image}",
                    "detail": "low"
                }
            }
        ]
    })

    while True:
        try:
            response = client.chat.completions.create(
                model=args.gpt_model_version,
                messages=local_prompt
            )
            break
        except Exception as e:
            tqdm.write(f"[Error - verify_image] {e}")
            time.sleep(0.1)
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue

    local_prompt.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response.choices[0].message.content
            }
        ]
    })

    verification = response.choices[0].message.content
    tqdm.write(f"[Result - verify_image] {verification}")
    if verification.lower().startswith('yes'):
        return local_prompt
    else:
        return None

def verify_hypothesis(args, prompt, generated_image, hypothesis, label_text, style_text):
    error_counter = 0
    local_prompt = copy.deepcopy(prompt)

    base64_generated_image = convert_to_base64(generated_image.resize((args.image_resize_size, args.image_resize_size)))
    local_prompt.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please verify if given hypothesis pair and its label is correct for the generated {style_text} style image. Start the response with 'Yes' or 'No'.\n\
Hypothesis: {hypothesis}\n\
Label: {label_text}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_generated_image}",
                    "detail": "low"
                }
            }
        ]
    })

    while True:
        try:
            response = client.chat.completions.create(
                model=args.gpt_model_version,
                messages=local_prompt
            )
            break
        except Exception as e:
            tqdm.write(f"[Error - verify_hypothesis] {e}")
            time.sleep(0.1)
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue

    ve_verification = response.choices[0].message.content
    tqdm.write(f"[Result - verify_hypothesis] {ve_verification}")
    if ve_verification.lower().startswith('yes'):
        return True, local_prompt
    else:
        return False, local_prompt

def generate_paraphrased_hypothesis(args, prompt, generated_image, hypothesis, style_text):
    error_counter = 0
    local_prompt = copy.deepcopy(prompt)

    base64_generated_image = convert_to_base64(generated_image.resize((args.image_resize_size, args.image_resize_size)))
    local_prompt.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please paraphrase the hypothesis sentence below for the generated {style_text} style image. The paraphrased hypothesis should have the same meaning as the original sentence but be rephrased in a different way. Only the sentence should be paraphrased.\n\
Hypothesis: {hypothesis}"
            },
        ]
    })

    while True:
        try:
            response = client.chat.completions.create(
                model=args.gpt_model_version,
                messages=local_prompt
            )
            paraphrased_hypothesis = response.choices[0].message.content
            try:
                paraphrased_hypothesis = paraphrased_hypothesis.split('Paraphrased Hypothesis: ')[-1].strip()
            except:
                # try again
                continue
            break
        except Exception as e:
            tqdm.write(f"[Error - generate_paraphrased_hypothesis] {e}")
            time.sleep(0.1)
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue

    tqdm.write(f"[Result - generate_paraphrased_hypothesis] {paraphrased_hypothesis}")
    return paraphrased_hypothesis, prompt

def annotate_entailment(args, prompt, generated_image, hypothesis):
    error_counter = 0
    local_prompt = copy.deepcopy(prompt)

    base64_generated_image = convert_to_base64(generated_image.resize((args.image_resize_size, args.image_resize_size)))
    local_prompt.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Does the given hypothesis entail the image? Start the response with 'True', 'False', or 'Undetermined'.\n\
Hypothesis: {hypothesis}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_generated_image}",
                    "detail": "low"
                }
            }
        ]
    })

    while True:
        try:
            response = client.chat.completions.create(
                model=args.gpt_model_version,
                messages=local_prompt
            )
            ve_correction = response.choices[0].message.content

            if ve_correction.lower().startswith('true'):
                ve_correction = 0
            elif ve_correction.lower().startswith('false'):
                ve_correction = 2
            elif ve_correction.lower().startswith('undetermined'):
                ve_correction = 1
            else:
                return None

            break
        except Exception as e:
            tqdm.write(f"[Error - annotate_entailment] {e}")
            time.sleep(0.1)
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue

    tqdm.write(f"[Result - annotate_entailment] {ve_correction}")
    return ve_correction, local_prompt

def build_default_prompt(args):
    base64_example1 = encode_image_base64('./task/visual_entailment/examples/example1.jpg')
    base64_output1 = encode_image_base64(f'./task/visual_entailment/examples/output1_{args.test_dataset}.jpg')

    # if args.test_dataset == 'snli_ve_sports_cartoon':
    #     args.target_domain_id = 1
    # elif args.test_dataset == 'snli_ve_sports_pencil':
    #     args.target_domain_id = 2
    # elif args.test_dataset == 'snli_ve_sports_oil':
    #     args.target_domain_id = 3

    messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an annotator for visual entailment tasks. You will help create stylized image and its corresponding hypothesis based on user requests."
            }
        ]
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Please generate a detailed prompt for DALL-E3 model to replicate the given image."
            },
            # {
            #     "type": "image_url",
            #     "image_url": {
            #         "url": f"data:image/jpeg;base64,{base64_example1}",
            #         "detail": "low"
            #     }
            # }
        ]
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Create an image of a man preparing food outside an industrial-style workspace. The man is wearing a flat cap and a dark short-sleeve shirt and is standing at a brown counter, chopping green onions on a cutting board. Surrounding him on the counter are various fresh vegetables, including green onions, leafy greens, a whole avocado, and a bowl of eggs. In the background, an open garage door reveals the interior of the workspace with tools, a workbench, and a bicycle leaning against the outside. The floor is concrete and the walls are decorated with hanging tools and shelves. The overall atmosphere should convey a casual, industrious vibe."
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please modify the generated prompt to change the style of the image to a {domain_map[args.target_domain_id]} style."
            }
        ]
    },
    ]

    if args.target_domain_id == 1:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Create a cartoon-style image of a man preparing food outside an industrial-style workspace. The man is wearing a flat cap and a dark short-sleeve shirt and is standing at a brown counter, chopping green onions on a cutting board. Surrounding him on the counter are various fresh vegetables, including green onions, leafy greens, a whole avocado, and a bowl of eggs. In the background, an open garage door reveals the interior of the workspace with tools, a workbench, and a bicycle leaning against the outside. The floor is concrete and the walls are decorated with hanging tools and shelves. The overall atmosphere should convey a casual, industrious vibe, with cartoonish exaggerated features and vibrant colors."
                }
            ]
        })
    elif args.target_domain_id == 2:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Create a pencil drawing of a man preparing food outside an industrial-style workspace. The man is wearing a flat cap and a dark short-sleeve shirt, standing at a brown counter, chopping green onions on a cutting board. Surrounding him on the counter are various fresh vegetables, including green onions, leafy greens, a whole avocado, and a bowl of eggs. In the background, an open garage door reveals the interior of the workspace with tools, a workbench, and a bicycle leaning against the outside. The floor is concrete, and the walls are decorated with hanging tools and shelves. The overall atmosphere should convey a casual, industrious vibe, with detailed cross-hatching and shading to enhance the pencil drawing style."
                }
            ]
        })
    elif args.target_domain_id == 3:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Create an oil painting of a man preparing food outside an industrial-style workspace. The man, wearing a flat cap and a dark short-sleeve shirt, is depicted standing at a brown counter, chopping green onions on a cutting board. The counter is adorned with various fresh vegetables, including green onions, leafy greens, a whole avocado, and a bowl of eggs, all rendered with thick, textured brushstrokes characteristic of oil painting. In the background, through the open garage door, details of the workspace interior are captured with an artistic flair, including tools, a workbench, and a bicycle leaning against the outside wall. The concrete floor and walls, decorated with hanging tools and shelves, are stylized with rich, vivid colors. The overall composition should evoke a casual, industrious atmosphere, with the warmth and depth typical of an oil painting."
                }
            ]
        })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please verify if the image below is a {domain_map[args.target_domain_id]} style image of the original image. The generated image should not exactly match the original image but should capture the essence of the original image. Start the response with 'Yes' or 'No'."
            },
            # {
            #     "type": "image_url",
            #     "image_url": {
            #         "url": f"data:image/jpeg;base64,{base64_output1}",
            #         "detail": "low"
            #     }
            # }
        ]
    })

    if args.target_domain_id == 1:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Yes, the provided image is a cartoon-style representation of the original image. The soccer player is depicted in a stylized, animated manner, with exaggerated features typical of cartoons. The attire, including the white short-sleeved jersey, blue shorts, long white socks, and white and orange cleats, closely matches the original image. The animated background with grass, orange cones, and a goal net also retains elements from the original setting, demonstrating a colorful and whimsical portrayal."
                }
            ]
        })
    elif args.target_domain_id == 2:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Yes, the image below is a pencil drawing style representation of the original image. The detail is rendered with fine lines and cross-hatching, capturing the man preparing food with various vegetables on the counter, the industrial background with tools, and the bicycle leaning against the outside, all depicted in a pencil sketch style."
                }
            ]
        })
    elif args.target_domain_id == 3:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Yes, the image below captures the requested modifications. It is an oil painting style rendition of the original image. The man is depicted chopping green onions on a counter, surrounded by fresh vegetables including green onions, leafy greens, and various other items. The background maintains the industrial workspace setting, featuring tools, a workbench, and a bicycle. The overall scene is characterized by the rich textures and vibrant colors typical of an oil painting."
                }
            ]
        })

    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please verify if given hypothesis pair and its label is correct for the generated {domain_map[args.target_domain_id]} style image. Start the response with 'Yes' or 'No'.\n\
Hypothesis: The person is preparing ingredients for a meal in an outdoor kitchen setup.\n\
Label: True"
            }
        ]
    })
    messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Yes, the hypothesis “The person is preparing ingredients for a meal in an outdoor kitchen setup” is entailed by the given image. The person is indeed chopping vegetables, and the setup appears to be an outdoor or semi-outdoor kitchen area."
            }
        ]
    })
    messages.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": f"Please paraphrase the hypothesis sentence below for the generated {domain_map[args.target_domain_id]} style image. The paraphrased hypothesis should have the same meaning as the original sentence but be rephrased in a different way. Only the sentence should be paraphrased.\n\
Hypothesis: The person is preparing ingredients for a meal in an outdoor kitchen setup."
            }
        ]
    })
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Paraphrased Hypothesis: The individual is getting ingredients ready for cooking in an outdoor kitchen setting."
            }
        ]
    })

    return messages
