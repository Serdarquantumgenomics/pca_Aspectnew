#!/bin/bash
#SBATCH -c 1
#SBATCH -t 16:20:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=130000
#SBATCH -o hostname_%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e hostname_%j.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=ALL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=serdar.aslan@gmail.com   # Email to which notifications will be sent

 
source tensorflow/bin/activate

module load gcc/6.2.0
module load cuda/9.0
module load python/3.6.0


python main5.py
#python main2.py
#python main2_knn.py
#python main0.py
#python main1.py
#python error2.py
#python Stage_1_Delay.py
#python Stage_2_Delay.py
#python Stage_2_Delay_eigenvectors_2.py

#python sobel3.py
