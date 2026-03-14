set -euo pipefail

ROOT=/root/autoresearch-macos-hrm
RESULTS_PATH=$ROOT/results.tsv
PROMISING_MARGIN=${PROMISING_MARGIN:-0.05}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

TARGET_EPOCH=$(date -u -d 'next monday 00:00' +%s)

parent_cfg() {
  python - <<'PY'
import json, os, pathlib
root = pathlib.Path("/root/autoresearch-macos-hrm")
elite_path = root / "elite.json"
role = os.environ.get("ROLE", "best")
elite = {}
if elite_path.exists():
    try:
        elite = json.loads(elite_path.read_text(encoding="utf-8"))
    except Exception:
        elite = {}
def pick(key):
    rec = elite.get(key)
    cfg = (rec or {}).get("config")
    if isinstance(cfg, dict) and cfg:
        return cfg
    return None
cfg = pick("best")
if role == "second":
    cfg = pick("second") or cfg
elif role == "promising":
    cfg = pick("promising") or cfg
elif role == "mix":
    cfg = pick("second") or pick("promising") or cfg
if cfg is None:
    cfg = {
      "depth": 8,
      "head_dim": 32,
      "aspect_ratio": 64,
      "h_cycles": 2,
      "l_cycles": 2,
      "forward_dtype": "bfloat16",
      "device_bs": 2,
      "total_bs": 2**16,
      "embedding_lr": 0.3,
      "matrix_lr": 0.04,
      "unembedding_lr": 0.004,
      "weight_decay": 0.15,
      "warmup_ratio": 0.02,
      "warmdown_ratio": 0.5,
      "final_lr_frac": 0.0,
      "logits_softcap": 15.0,
      "zero_o_proj_init": 1,
    }
print(json.dumps(cfg, ensure_ascii=False))
PY
}

mutate_env() {
  python - <<'PY'
import json, os, random, math
cfg = json.loads(os.environ["PARENT_CFG"])
rng = random.Random(int(os.environ["SEED"]))
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x
def lnorm(center, sigma, lo, hi):
    center = max(float(center), 1e-6)
    v = rng.lognormvariate(math.log(center), sigma)
    return clamp(v, lo, hi)
def norm(center, sigma, lo, hi):
    v = rng.gauss(float(center), sigma)
    return clamp(v, lo, hi)
arch = ["depth","head_dim","aspect_ratio","h_cycles","l_cycles","logits_softcap","zero_o_proj_init","forward_dtype"]
opt = ["embedding_lr","matrix_lr","unembedding_lr","weight_decay","warmup_ratio","warmdown_ratio","final_lr_frac","device_bs","total_bs"]
k = rng.choice(arch if rng.random() < 0.7 else opt)
ks = [k]
if rng.random() < 0.2:
    ks.append(rng.choice([x for x in (arch+opt) if x != k]))
for key in ks:
    if key == "aspect_ratio":
        cfg[key] = int(clamp(int(round(norm(cfg.get(key,64), 8, 16, 128))), 16, 128))
    elif key == "head_dim":
        cfg[key] = rng.choice([16,32,64])
    elif key == "depth":
        cfg[key] = rng.choice([6,8,10,12])
    elif key == "h_cycles":
        cfg[key] = rng.choice([1,2,3])
    elif key == "l_cycles":
        cfg[key] = rng.choice([1,2,3])
    elif key == "logits_softcap":
        cfg[key] = rng.choice([0.0,10.0,15.0,20.0,30.0])
    elif key == "zero_o_proj_init":
        cfg[key] = rng.choice([0,1])
    elif key == "forward_dtype":
        cfg[key] = rng.choice(["float32","bfloat16"])
    elif key == "embedding_lr":
        cfg[key] = lnorm(cfg.get(key,0.3), 0.18, 0.02, 1.2)
    elif key == "matrix_lr":
        cfg[key] = lnorm(cfg.get(key,0.04), 0.18, 0.002, 0.12)
    elif key == "unembedding_lr":
        cfg[key] = lnorm(cfg.get(key,0.004), 0.18, 0.0005, 0.02)
    elif key == "weight_decay":
        cfg[key] = norm(cfg.get(key,0.15), 0.03, 0.0, 0.3)
    elif key == "warmup_ratio":
        cfg[key] = rng.choice([0.0,0.02,0.05,0.1])
    elif key == "warmdown_ratio":
        cfg[key] = rng.choice([0.3,0.5,0.7])
    elif key == "final_lr_frac":
        cfg[key] = rng.choice([0.0,0.05])
    elif key == "device_bs":
        cfg[key] = rng.choice([1,2,4])
    elif key == "total_bs":
        cfg[key] = rng.choice([2**14,2**15,2**16,2**17])
if cfg.get("forward_dtype") not in ("float32","bfloat16"):
    cfg["forward_dtype"] = "bfloat16"
env = {
  "DEPTH": int(cfg.get("depth",8)),
  "HEAD_DIM": int(cfg.get("head_dim",32)),
  "ASPECT_RATIO": int(cfg.get("aspect_ratio",64)),
  "H_CYCLES": int(cfg.get("h_cycles",2)),
  "L_CYCLES": int(cfg.get("l_cycles",2)),
  "FORWARD_DTYPE": str(cfg.get("forward_dtype","bfloat16")),
  "PRENORM": int(cfg.get("prenorm",0)),
  "LAYER_SCALE_INIT": float(cfg.get("layer_scale_init",0.0)),
  "DEVICE_BATCH_SIZE": int(cfg.get("device_bs",2)),
  "TOTAL_BATCH_SIZE": int(cfg.get("total_bs",2**16)),
  "EMBEDDING_LR": float(cfg.get("embedding_lr",0.3)),
  "MATRIX_LR": float(cfg.get("matrix_lr",0.04)),
  "UNEMBEDDING_LR": float(cfg.get("unembedding_lr",0.004)),
  "WEIGHT_DECAY": float(cfg.get("weight_decay",0.15)),
  "WARMUP_RATIO": float(cfg.get("warmup_ratio",0.02)),
  "WARMDOWN_RATIO": float(cfg.get("warmdown_ratio",0.5)),
  "FINAL_LR_FRAC": float(cfg.get("final_lr_frac",0.0)),
  "LOGITS_SOFTCAP": float(cfg.get("logits_softcap",15.0)),
  "ZERO_O_PROJ_INIT": int(cfg.get("zero_o_proj_init",1)),
}
print(" ".join(f"{k}={env[k]}" for k in env.keys()))
PY
}

run_one() {
  GPU=$1
  WT=$2
  ROLE=$3
  i=$4
  SEED=$5
  export ROLE SEED
  PARENT_CFG=$(ROLE="$ROLE" parent_cfg)
  EXP_ENV=$(PARENT_CFG="$PARENT_CFG" SEED="$SEED" mutate_env)
  EXP_DESC_STR="gpu${GPU} ${ROLE} run${i} ${EXP_ENV}"
  cd "$WT"
  mkdir -p logs
  env HF_ENDPOINT="$HF_ENDPOINT" RESULTS_PATH="$RESULTS_PATH" CUDA_VISIBLE_DEVICES="$GPU" SEED="$SEED" PROMISING_MARGIN="$PROMISING_MARGIN" "EXP_DESC=$EXP_DESC_STR" $EXP_ENV \
    python train.py > "logs/run_${GPU}_${i}.log" 2>&1 || true
}

while true; do
  NOW_EPOCH=$(date -u +%s)
  if [ "$NOW_EPOCH" -ge "$TARGET_EPOCH" ]; then
    exit 0
  fi
  while pgrep -f "python train.py" >/dev/null 2>&1; do
    sleep 30
  done
  RUN_ID="$NOW_EPOCH"
  run_one 0 /root/autoresearch-macos-hrm-wt-gpu0 best "$RUN_ID" $((0*100000 + NOW_EPOCH)) &
  run_one 1 /root/autoresearch-macos-hrm-wt-gpu1 second "$RUN_ID" $((1*100000 + NOW_EPOCH)) &
  run_one 2 /root/autoresearch-macos-hrm-wt-gpu2 promising "$RUN_ID" $((2*100000 + NOW_EPOCH)) &
  run_one 3 /root/autoresearch-macos-hrm-wt-gpu3 mix "$RUN_ID" $((3*100000 + NOW_EPOCH)) &
  wait
done
