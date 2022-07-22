#SBATCH --job-name="<JOB_NAME>"
#SBATCH -D .
#SBATCH --output=./logs/<LOG_NAME>_%j.out
#SBATCH --error=./logs/<LOG_NAME>_%j.err
#SBATCH --gres=gpu:<GPUS_PER_NODE>
#SBATCH --cpus-per-task=<CPUS_PER_NODE>
#SBATCH --nodes=<NUM_NODES>
#SBATCH --time=<TIME>

# source venv and module load
if uname -a | grep -q amd
then
	module load cmake/3.18.2 gcc/10.2.0 rocm/5.1.1 mkl/2018.4 intel/2018.4 python/3.7.4
	source ../../venv/bin/activate
	export LD_LIBRARY_PATH=../../utils/external-lib:$LD_LIBRARY_PATH
elif uname -a | grep -q p9
then
	module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 openmpi/4.0.1 \
					atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 \
					python/3.7.4_ML arrow/3.0.0 text-mining/2.0.0 torch/1.9.0a0 torchvision/0.11.0
else
	source ../../venv/bin/activate
fi
