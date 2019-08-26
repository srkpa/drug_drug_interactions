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
	    -e|--exptname)
	    expt_name="$2"
	    shift # past argument
	    shift # past value
	    ;;
	    -i|--instance)
	    instance_type="$2"
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

server=${server:=local}
instance_type=${instance_type:=c4.2xlarge}

#python ${expt_name}/config.py -o ${expt_name}/configs.json
case ${server} in
  aws)
    exp_launch --launch-config launch_settings.json --exp-config ${expt_name}/configs.json --exp-name ${expt_name} -t $instance_type --user-data-path user_data.txt
    ;;

  local)
    out=../results/${expt_name}
    echo "Storing experiment results in ${out}"

    mkdir -p ${out}
    dispatcher run -o ${out} -p ${expt_name}/configs.json -E ../bin/train
    ;;

  mcg)
   mcguffin launch --exp-name ${expt_name} --launch-config launch_settings.json  --exp-config ${expt_name}/configs.json --save-checkpoint
   ;;

  *)
    echo "Mode is invalid. Options are debug or prod."
    exit 1
    ;;
esac