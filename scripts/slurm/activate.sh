# Sourced by every sbatch script. Puts the venv on PATH and sanity-checks it.
#
# Deliberately does NOT rely on `source .venv/bin/activate` taking effect. That
# script is written for interactive shells and has proved unreliable under
# `set -euo pipefail` in a batch job (it failed with "python: command not found",
# i.e. it returned without having modified PATH). Prepending the venv's bin
# directory ourselves is one line and cannot half-succeed.
#
# Module names come from .venv/ada_modules, written by setup_env.sh, rather than
# being hard-coded: Ada's module names drift, the wiki's list is out of date, and
# the venv's bin/python symlinks to whichever interpreter created it -- so a job
# must reproduce the same modules or the venv will not run. On the current
# cluster no modules are needed at all and that file is legitimately empty.

if [ -f .venv/ada_modules ]; then
  # Skip blanks and a bare `module load` with no argument -- an earlier bug wrote
  # exactly that, and `module load` with no operand is an error.
  while read -r line; do
    case "$line" in
      ""|"module load"|"module load ") continue ;;
    esac
    eval "$line" 2>/dev/null || true
  done < .venv/ada_modules
fi

VENV="$PWD/.venv"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
unset PYTHONHOME

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [ ! -x "$VENV/bin/python" ]; then
  echo "FATAL: $VENV/bin/python is missing or not executable."
  echo "Re-run scripts/slurm/setup_env.sh on the login node."
  ls -l "$VENV/bin/" 2>&1 | head -20
  exit 1
fi

if ! "$VENV/bin/python" - <<'EOF'
import sys
import torch, transformers
print(f"  python {sys.version.split()[0]} at {sys.executable}")
print(f"  torch {torch.__version__}  transformers {transformers.__version__}  "
      f"cuda={torch.cuda.is_available()}")
EOF
then
  echo
  echo "FATAL: the venv exists but cannot import its packages."
  echo "Diagnostics:"
  echo "  node:        $(hostname)"
  echo "  cwd:         $PWD"
  echo "  PATH:        $PATH"
  echo "  which python: $(command -v python || echo '<none>')"
  echo "  venv python:  $(ls -l "$VENV/bin/python" 2>&1)"
  echo "  pyvenv.cfg:"
  sed 's/^/    /' "$VENV/pyvenv.cfg" 2>&1
  echo "  base interpreter exists: $(ls -l "$(sed -n 's/^home = //p' "$VENV/pyvenv.cfg")/python3" 2>&1)"
  echo "  site-packages:"
  ls -d "$VENV"/lib/python*/site-packages 2>&1 | sed 's/^/    /'
  exit 1
fi
