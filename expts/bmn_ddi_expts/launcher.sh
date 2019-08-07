#!/bin/bash
#optional parameters
while [[ $# -gt 0 ]]
do
	case "$1" in
	    -s|--server)
	    server="$2"
	    shift # past argument
	    shift # past value
	    ;;
	    *)
		shift
		echo "Invalid option -$1" >&2
	    ;;
	esac
done
set -e

#server={server:-local}
server=${server:=local}
name=bmn_expts
instance_type="c5.2xlarge"

python config.py -o configs.json
case ${server} in
  aws)
    exp_launch --launch-config launch_settings.json --exp-config configs.json --exp-name ${name} -t $instance_type --user-data-path user_data.txt
    ;;

  local)
    out=../../results/$(basename $(pwd))
    echo "Storing experiment results in ${out}"

    mkdir -p ${out}
    dispatcher run -o ${out} -p configs.json -E ../../bin/train
    ;;
  *)
    echo "Mode is invalid. Options are debug or prod."
    exit 1
    ;;
esac