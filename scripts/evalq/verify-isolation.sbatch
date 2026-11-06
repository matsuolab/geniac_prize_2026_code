#!/bin/bash
# Prove, on this cluster's actual compute nodes, that a submission container is
# cut off. Run it before a competition round and after any Singularity or driver
# change — the isolation is a property of the runtime, not of our code, and it is
# the one thing a bad upgrade could silently take away.
#
#   sbatch -p part-$(id -gn) --gres=gpu:1 scripts/evalq/verify-isolation.sbatch
#
# Checks, in the exact shape capx.evalqueue.runner uses:
#   1. --net --network=none leaves a loopback interface and no route off the node
#   2. --containall hides the host filesystem except explicit binds: neither the
#      backend socket beside the bound gateway directory nor the checkout is visible.
#      The same look without --containall must see both, or the check proves nothing.
#   3. an AF_UNIX socket still reaches an organizer process outside that namespace
#   4. --nv still injects the GPU
#SBATCH --job-name=capx-verify-isolation
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x.%j.log

set -uo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs
if ! command -v singularity >/dev/null; then
    # The system profile sets up `module` whatever environment the job got, without
    # anyone's login files. It does not run under -e or -u.
    set +u
    # shellcheck disable=SC1091
    source /etc/profile >/dev/null 2>&1
    set -u
    module load "${CAPX_SINGULARITY_MODULE:-singularitypro/4.1}"
fi

IMAGE="${1:-singularity/images/capx-submission-agent0.sif}"
if [[ ! -f "${IMAGE}" ]]; then
    echo "FAIL: no image at ${IMAGE} (build it with singularity/build.sbatch)" >&2
    exit 1
fi

RUN="$(mktemp -d /tmp/capx-verify.XXXX)"
mkdir -p "${RUN}/gw" "${RUN}/be"
trap 'rm -rf "${RUN}"' EXIT
failures=0
check() { if [[ "$2" == "$3" ]]; then echo "  ok   $1"; else echo "  FAIL $1 (got '$2', want '$3')"; failures=$((failures + 1)); fi; }

echo "node=$(hostname) image=${IMAGE}"

# An organizer-side listener on a unix socket, standing in for the gateway.
python3 - "${RUN}/gw/gateway.sock" <<'PY' &
import os, socket, sys
path = sys.argv[1]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(path); s.listen(); os.chmod(path, 0o600)
conn, _ = s.accept()
conn.sendall(b"pong:" + conn.recv(64)); conn.close()
PY
LISTENER=$!
for _ in $(seq 40); do [[ -S "${RUN}/gw/gateway.sock" ]] && break; sleep 0.25; done
# A backend socket the submission must NOT be able to reach, by its host path.
touch "${RUN}/be/robosuite.sock"
HOST_PATHS="${RUN}/be/robosuite.sock:${PWD}/scripts/evalq/verify-isolation.sbatch"

# Looks for the host paths from inside a container started with the given flags.
host_paths_seen() {
    singularity exec --net --network=none "$@" --pwd /workspace/capx \
        --bind "${RUN}/gw:/capx/run:ro" "${IMAGE}" \
        python -c 'import os, sys; print(sum(map(os.path.exists, sys.argv[1].split(":"))))' \
        "${HOST_PATHS}" 2>&1 | tail -n 1
}

echo "running the submission-shaped container"
OUT="$(singularity exec \
    --net --network=none --containall --nv \
    --pwd /workspace/capx \
    --bind "${RUN}/gw:/capx/run:ro" \
    "${IMAGE}" python - <<'PY' 2>&1
import os, socket, subprocess
# bonding_masters is a sysfs control file, not an interface.
ifaces = sorted(n for n in os.listdir("/sys/class/net") if n != "bonding_masters")
print("IFACES", ",".join(ifaces))
try:
    socket.create_connection(("1.1.1.1", 443), timeout=5)
    print("EGRESS reachable")
except OSError:
    print("EGRESS blocked")
print("GATEWAY_VISIBLE", os.path.exists("/capx/run/gateway.sock"))
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/capx/run/gateway.sock")
s.sendall(b"ping")
print("UNIX_RPC", s.recv(64).decode())
print("GPU", subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU "))
PY
)"
wait "${LISTENER}" 2>/dev/null
HIDDEN="$(host_paths_seen --containall)"
CONTROL="$(host_paths_seen)"

echo "${OUT}" | sed 's/^/    /'
echo
echo "results:"
check "only loopback in the container"  "$(echo "${OUT}" | awk '/^IFACES/{print $2}')" "lo"
check "no egress"                       "$(echo "${OUT}" | awk '/^EGRESS/{print $2}')" "blocked"
check "host paths hidden"               "${HIDDEN}" "0"
check "control: seen without --containall" "${CONTROL}" "2"
check "gateway socket reachable"        "$(echo "${OUT}" | awk '/^GATEWAY_VISIBLE/{print $2}')" "True"
check "gateway answers over AF_UNIX"    "$(echo "${OUT}" | awk '/^UNIX_RPC/{print $2}')" "pong:ping"
check "one GPU injected"                "$(echo "${OUT}" | awk '/^GPU/{print $2}')" "1"

echo
if (( failures )); then
    echo "ISOLATION CHECK FAILED (${failures})"
    exit 1
fi
echo "isolation verified"
