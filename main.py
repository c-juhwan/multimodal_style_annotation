# Standard Library Modules
import time
import argparse
# Custom Modules
from utils.arguments import ArgParser
from utils.utils import check_path, set_random_seed

def main(args: argparse.Namespace) -> None:
    # Set random seed
    if args.seed not in [None, 'None']:
        set_random_seed(args.seed)

    start_time = time.time()

    # Check if the path exists
    for path in []:
        check_path(path)

    # Get the job to do
    if args.job == None:
        raise ValueError('Please specify the job to do.')
    else:
        if args.task == 'captioning':
            if args.job == 'preprocessing':
                from task.captioning.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                    from task.captioning.train import training as job
            elif args.job == 'testing':
                from task.captioning.test import testing as job
            elif args.job == 'annotating':
                from task.captioning.annotating import annotating_captioning as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'visual_qa':
            if args.job == 'preprocessing':
                from task.visual_qa.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.visual_qa.train import training as job
            elif args.job == 'testing':
                from task.visual_qa.test import testing as job
            elif args.job == 'annotating':
                from task.visual_qa.annotating2 import annotating_vqa as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        elif args.task == 'visual_entailment':
            if args.job == 'preprocessing':
                from task.visual_entailment.preprocessing import preprocessing as job
            elif args.job in ['training', 'resume_training']:
                from task.visual_entailment.train import training as job
            elif args.job == 'testing':
                from task.visual_entailment.test import testing as job
            elif args.job == 'annotating':
                from task.visual_entailment.annotating import annotating_ve as job
            else:
                raise ValueError(f'Invalid job: {args.job}')
        else:
            raise ValueError(f'Invalid task: {args.task}')

    # Do the job
    job(args)

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time / 60:.2f} minutes')

if __name__ == '__main__':
    # Parse arguments
    parser = ArgParser()
    args = parser.get_args()

    if len(args.task_dataset) == 1:
        args.task_dataset = args.task_dataset[0]

    # Run the main function
    main(args)
