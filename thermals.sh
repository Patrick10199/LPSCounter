#!/bin/bash
# CPU thermal/power/throttle telemetry for the search box
echo "=== lm_sensors ==="
if command -v sensors >/dev/null; then
  sensors 2>/dev/null | grep -iE 'package|core [0-9]|pkg|temp' | head -30
else
  echo "sensors not installed"
fi
echo "=== coretemp thermal zones ==="
for z in /sys/class/thermal/thermal_zone*; do
  [ -f "$z/temp" ] || continue
  t=$(cat "$z/temp")
  type=$(cat "$z/type" 2>/dev/null)
  echo "$type: $((t/1000)) C"
done
echo "=== hwmon coretemp ==="
for h in /sys/class/hwmon/hwmon*; do
  name=$(cat "$h/name" 2>/dev/null)
  [ "$name" = "coretemp" ] || continue
  for f in "$h"/temp*_input; do
    [ -f "$f" ] || continue
    lbl=$(cat "${f%_input}_label" 2>/dev/null)
    echo "$lbl: $(($(cat "$f")/1000)) C"
  done
done
echo "=== throttle counters (thermal_throttle) ==="
tot=0
for c in /sys/devices/system/cpu/cpu*/thermal_throttle/core_throttle_count; do
  [ -f "$c" ] || continue
  v=$(cat "$c")
  tot=$((tot+v))
done
echo "summed core_throttle_count across all cores: $tot"
echo "=== package power (RAPL, if exposed) ==="
for p in /sys/class/powercap/intel-rapl:0; do
  [ -d "$p" ] || continue
  e1=$(cat "$p/energy_uj" 2>/dev/null); sleep 1; e2=$(cat "$p/energy_uj" 2>/dev/null)
  [ -n "$e1" ] && [ -n "$e2" ] && echo "package draw: $(( (e2-e1)/1000000 )) W (constraint: $(cat "$p/constraint_0_power_limit_uw" 2>/dev/null | awk '{print $1/1000000" W"}'))"
done
echo "=== dmesg thermal/mce (last 5) ==="
dmesg 2>/dev/null | grep -iE 'thermal|throttl|mce|hardware error' | tail -5
echo "(none above = clean)"
