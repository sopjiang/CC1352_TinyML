#!/bin/bash
set -e


SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
echo ""

echo "Flashing board... "
echo ""

dslite.sh -c ${SCRIPTPATH}/user_files/configs/cc1352p1f3.ccxml -l ${SCRIPTPATH}/user_files/settings/generated.ufsettings -e -f -v ${SCRIPTPATH}/edge_impulse_firmware.out

echo ""
echo "Flashed your TI LaunchXL development board."
echo "To set up your development with Edge Impulse, run 'edge-impulse-daemon'"
echo "To run your impulse on your development board, run 'edge-impulse-run-impulse'"
