#!/bin/bash
#SBATCH --job-name=cfr555  # 作业名称（与你的-J参数一致）
#SBATCH --output=SLURM_Logs/%j.out                    # 标准输出日志路径（%j会被替换为作业ID）
#SBATCH --error=SLURM_Logs/%j.err                     # 错误输出日志路径
#SBATCH --partition=a100
#SBATCH -w compute-a30-01,compute-a100-01,compute-a100-02,compute-a100-02,compute-a100-04                              # 使用a100分区
#SBATCH --gres=gpu:1                                  # 每节点申请1块GPU
#SBATCH --nodes=1                                     # 申请1个计算节点
#SBATCH --ntasks-per-node=1                           # 每节点运行1个任务（推荐显式声明）
#SBATCH --cpus-per-task=8                             # 建议为每个任务分配CPU核心（根据需求调整）
#SBATCH --mem-per-cpu=8GB


source /home/sxzhuangch/miniconda3/etc/profile.d/conda.sh    # 替换为你的conda安装路径
conda activate zhh

export PYTHONUNBUFFERED=1

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: Allocated GPU not accessible."
    exit 1
fi

echo "GPU check passed."

for mode in fixed_potential ; do
    echo "Running mode=${mode}"
    python3 -u -m main \
            --reward_mode ${mode}
done