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

def annotating_captioning(args):
    assert args.task_dataset in ['uit_viic_en_ori'], "Only support original datasets as a source to annotate."
    assert args.test_dataset in ['uit_viic_en_cartoon', 'uit_viic_en_pencil', 'uit_viic_en_oil'], "Only support cartoon, pencil, and oil painting as target domains."

    if args.test_dataset == 'uit_viic_en_cartoon':
        args.target_domain_id = 1
    elif args.test_dataset == 'uit_viic_en_pencil':
        args.target_domain_id = 2
    elif args.test_dataset == 'uit_viic_en_oil':
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
            'caption': train_ori_data['caption'],
            'caption_number': train_ori_data['caption_number'],
            'all_captions': train_ori_data['all_captions'],
            'domain_id': train_ori_data['domain_id'],
        },
        'valid': {
            'image': valid_ori_data['image'],
            'image_id': valid_ori_data['image_id'],
            'caption': valid_ori_data['caption'],
            'caption_number': valid_ori_data['caption_number'],
            'all_captions': valid_ori_data['all_captions'],
            'domain_id': valid_ori_data['domain_id'],
        },
        'test': {
            'image': test_ori_data['image'],
            'image_id': test_ori_data['image_id'],
            'caption': test_ori_data['caption'],
            'caption_number': test_ori_data['caption_number'],
            'all_captions': test_ori_data['all_captions'],
            'domain_id': test_ori_data['domain_id'],
        }
    }

    # only collect caption_number == 0
    for split in ['valid', 'test', 'train']:
        only_data = {
            'image': [],
            'image_id': [],
            'caption': [],
            'caption_number': [],
            'all_captions': [],
            'domain_id': [],
        }
        for i in range(len(data_dict[split]['image'])):
            if data_dict[split]['caption_number'][i] == 0:
                only_data['image'].append(data_dict[split]['image'][i])
                only_data['image_id'].append(data_dict[split]['image_id'][i])
                only_data['caption'].append(data_dict[split]['caption'][i])
                only_data['caption_number'].append(data_dict[split]['caption_number'][i])
                only_data['all_captions'].append(data_dict[split]['all_captions'][i])
                only_data['domain_id'].append(data_dict[split]['domain_id'][i])
        data_dict[split] = only_data

    # Every element should have same length
    assert len(data_dict['train']['image']) == len(data_dict['train']['image_id']) == len(data_dict['train']['caption']) == len(data_dict['train']['caption_number']) == len(data_dict['train']['all_captions']) == len(data_dict['train']['domain_id'])
    assert len(data_dict['valid']['image']) == len(data_dict['valid']['image_id']) == len(data_dict['valid']['caption']) == len(data_dict['valid']['caption_number']) == len(data_dict['valid']['all_captions']) == len(data_dict['valid']['domain_id'])
    assert len(data_dict['test']['image']) == len(data_dict['test']['image_id']) == len(data_dict['test']['caption']) == len(data_dict['test']['caption_number']) == len(data_dict['test']['all_captions']) == len(data_dict['test']['domain_id'])
    print(len(list(set(data_dict['train']['image_id']))), len(list(set(data_dict['valid']['image_id']))), len(list(set(data_dict['test']['image_id']))))

    for split in ['valid', 'test', 'train']:
        new_data = {
            'image': [],
            'image_id': [],
            'caption': [],
            'caption_number': [],
            'all_captions': [],
            'domain_id': [],
            'original_prompt': [],
            'style_prompt': [],
        }

        print()

        for index in tqdm(range(len(data_dict[split]['image'])), desc=f'Annotating {split} data'):
            image = convert_to_base64(data_dict[split]['image'][index].resize((args.image_resize_size, args.image_resize_size)))
            prompt = build_default_prompt(args)

            # Captioning Annotation Step 1. Generate prompt for the given image
            original_prompt = generate_original_prompt(args, image, prompt)
            if original_prompt is not None:
                original_prompt, prompt = original_prompt # unpack the tuple
            else:
                tqdm.write(f'[FAILURE - generate_original_prompt] Error in generating original prompt. Skipping the index {index}...')
                index += 1
                continue

            # Captioning Annotation Step 2. Modify the generated prompt to change the style of the image
            style_prompt = generate_style_prompt(args, prompt)
            if style_prompt is not None:
                style_prompt, prompt = style_prompt # unpack the tuple
            else:
                tqdm.write(f'[FAILURE - generate_style_prompt] Error in generating style prompt. Skipping the index {index}...')
                index += 1
                continue

            # Captioning Annotation Step 3. Generate the image based on the modified prompt using DALL-E3 model
            # Captioning Annotation Step 4. Verify if the generated image is a {domain_map[args.target_domain_id]} style image of the original image
            generation_error_count = 0
            while True:
                generated_image = generate_image(args, style_prompt)
                if generated_image is None:
                    tqdm.write(f'[Error - generate_image] Error in generating image. Re-trying...')
                    break

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
            if generated_image is None:
                index += 1
                continue # move to the next image

            # Captioning Annotation Step 5. Generate five captions of the generated image
            generated_captions = generate_paraphrased_captions(args, prompt, data_dict[split]['all_captions'][index], style_prompt)
            if generated_captions is None:
                tqdm.write(f'[Error - generate_paraphrased_captions] Error in generating captions. Skipping the index {index}...')
                index += 1
                continue
            else:
                generated_captions, prompt = generated_captions # unpack the tuple
            tqdm.write(f'[Result - generate_paraphrased_captions] {str(generated_captions)}')

            # Captioning Annotation Step 6. Save the annotated data
            # Resize PIL image to 256x256 and save it
            for j in range(5):
                new_data['image'].append(generated_image.convert('RGB').resize((args.image_resize_size, args.image_resize_size)))
                new_data['image_id'].append(data_dict[split]['image_id'][index])
                new_data['caption'].append(generated_captions[j])
                new_data['caption_number'].append(j)
                new_data['all_captions'].append(generated_captions)
                new_data['domain_id'].append(args.target_domain_id)
                new_data['original_prompt'].append(original_prompt)
                new_data['style_prompt'].append(style_prompt)

            prompt = None # reset the prompt
            if (i > 0 and i % 10 == 0) or i == len(data_dict[split]['image']) - 1:
                with open(os.path.join(args.preprocess_path, args.test_dataset, f'{split}_temp.pkl'), 'wb') as f:
                    pickle.dump(new_data, f)
                    tqdm.write(f'[MESSAGE - Saving Temp Data] Saved {split} temp data')

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

def generate_paraphrased_captions(args, prompt, all_captions, style_text):
    error_counter = 0
    local_prompt = copy.deepcopy(prompt)

    local_prompt.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Please generate five captions of the generated {style_text} image. The captions should describe the image in different ways. Consider given captions below for reference.\n\n\
1. {all_captions[0]}\n\
2. {all_captions[1]}\n\
3. {all_captions[2]}\n\
4. {all_captions[3]}\n\
5. {all_captions[4]}"
            }
        ]
    })

    while True:
        try:
            response = client.chat.completions.create(
                model=args.gpt_model_version,
                messages=local_prompt
            )
        except Exception as e:
            tqdm.write(f"[Error - generate_paraphrased_captions] {e}")
            time.sleep(0.1)
            error_counter += 1
            if error_counter >= args.error_patience:
                return None
            continue

        try:
            generated_captions = [
                response.choices[0].message.content.split('\n')[i][3:] for i in range(5) # remove the number at the beginning of each caption
            ]
            break
        except Exception as e:
            tqdm.write(f"Error - generate_paraphrased_captions] {e}")
            time.sleep(0.5)
            continue # continue to generate captions until the annotator provides the correct format

    local_prompt.append({
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": response.choices[0].message.content
            }
        ]
    })

    return generated_captions, local_prompt

def build_default_prompt(args):
    base64_example1 = encode_image_base64('./task/captioning/examples/example1.jpg')
    base64_output1 = encode_image_base64(f'./task/captioning/examples/output1_{args.test_dataset}.jpg')

    # if args.test_dataset == 'uit_viic_en_cartoon':
    #     args.target_domain_id = 1
    # elif args.test_dataset == 'uit_viic_en_pencil':
    #     args.target_domain_id = 2
    # elif args.test_dataset == 'uit_viic_en_oil':
    #     args.target_domain_id = 3

    messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an annotator for image captioning tasks. You will help create stylized image and its captions based on user requests."
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
                "text": f"Please generate five captions of the generated {domain_map[args.target_domain_id]} style image. The captions should describe the image in different ways. Consider given captions below for reference.\n\n\
1. a man cutting up vegetables on top of a food cart.\n\
2. A man cutting up scallions at an outdoor table\n\
3. A man chopping vegetables on a white board.\n\
4. A man stands as he cuts up some vegetables.\n\
5. A man stands with a knife and onions in front of a garage"
            }
        ]
    })

    if args.target_domain_id == 1:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "1. A cartoon-styled man slicing vegetables at an outdoor table near a garage.\n\
2. An animated character chopping fresh scallions on a cutting board outside.\n\
3. A cartoon figure preparing a meal by cutting vegetables on a white board in front of a workshop.\n\
4. A man in cartoon form stands at an outdoor table, diligently cutting vegetables.\n\
5. A cartoon man, standing with a knife and spring onions, prepares food outside a garage filled with tools."
                }
            ]
        })
    elif args.target_domain_id == 2:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "1. A man skillfully slices green onions at a makeshift outdoor kitchen under a garage.\n\
2. A person in a flat cap concentrates on chopping fresh scallions at a rustic workstation.\n\
3. A man diligently prepares vegetables outside a workshop, surrounded by tools and bikes.\n\
4. In front of an open garage, a man methodically cuts vegetables on a table brimming with ingredients.\n\
5. A man stands at an improvised kitchen station, knife in hand, preparing fresh produce amidst an industrial backdrop."
                }
            ]
        })
    elif args.target_domain_id == 3:
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "1. A man skillfully chops scallions on a rustic wooden table outside an open garage.\n\
2. In an industrial setting, a man prepares fresh vegetables with precise knife cuts.\n\
3. An artistically rendered chef meticulously cuts green onions on a countertop, surrounded by various veggies.\n\
4. A bearded man in a flat cap slices vegetables at a makeshift outdoor kitchen, with a bicycle and tools in the background.\n\
5. Amidst an open workshop, a focused man stands, chopping green onions on a vibrant table filled with produce."
                }
            ]
        })

    return messages
