#!/bin/bash

# Example of usage:
# bash build_personal_image.sh -n imageXYZ -i content.txt
usage () {
    echo
    printf "Usage: %s [OPTIONS] <ARGS>\n
    This script builds a personal image from a starting invivobase image
    [OPTIONS] and corresponding <ARGS> are:\n
    [-n] <imagename> # Name of the docker image to build. You can provide it with a tag if you want
    [-b] <baseimage> # Name of the base image to use. 
         If -b is given without argument, the 'invivobase:latest' image will be used. 
         Therefore a <latest> tag should always be present for invivobase. Use multiple tag for that:
         ex: docker build -t invivobase:dev -t invivobase:latest
    [-e] <branchname> # The branch on invivobase that should be check into
    [-i] <inputfile> # Listing of the files/directory to include in your image
    [-r] # If the package list should be pruned to match the current list in the yaml file
    [-h] # Show this help, then exit
    \n" $(basename "$0") 1>&2; exit 1;
}


build(){
	account=$(aws sts get-caller-identity --query Account --output text)
	# Get the region defined in the current configuration (default to us-west-2 if none defined)
	region=$(aws configure get region)
	region=${region:-us-east-1}
	fullname="${account}.dkr.ecr.${region}.amazonaws.com/${input_imagename}"
	startpoint="${account}.dkr.ecr.${region}.amazonaws.com/${base_imagename}"
	# If the repository doesn't exist in ECR, create it.
	aws ecr describe-repositories --repository-names "${input_imagename}" > /dev/null 2>&1
	if [ $? -ne 0 ]
	then
	    aws ecr create-repository --repository-name "${input_imagename}" > /dev/null
	fi
	# Get the login command from ECR and execute it directly
	$(aws ecr get-login --region ${region} --no-include-email)
	
	content_dir=./content_${input_imagename}
	dockerfile=${content_dir}/Dockerfile
	mkdir -p ${content_dir}
	echo ${content_dir}
	if [ -r "$inputfile" ]; then
		cat ${inputfile} | while read f; do
		    cp -r $f ${content_dir}
		done
	fi
	# Build the docker image locally with the image name
	printf "FROM ${startpoint}\nCOPY $content_dir ./" > ${dockerfile}
	
	if [ -f "{$content_dir}/conda_env.yml" ]; then
		printf "\nRUN conda env update --file conda_env.yml ${replace}" >> ${dockerfile}	
	fi	
	#printf "\nRUN conda env create --name ${input_imagename} --file conda_env.yml " >> ${dockerfile}
	#printf "\nRUN sed -i '/conda/d' ~/.bashrc" >> ${dockerfile}
	#printf '\nRUN printf "\\n. /root/miniconda3/etc/profile.d/conda.sh;" >> ~/.bashrc' >> ${dockerfile}
	#printf "\nENV PATH=/root/miniconda3/envs/${input_imagename}/bin:\$PATH" >> ${dockerfile}
	#printf "\nRUN echo \"conda activate ${input_imagename}\"  >> ~/.bashrc && source ~/.bashrc" >> ${dockerfile}
	printf "\nRUN cd /app/invivobase && git checkout ${branch}  && pip install -e . && cd" >> ${dockerfile}
	printf "\nRUN . ~/.bashrc" >> ${dockerfile}
	printf "\nRUN pip install -e ." >> ${dockerfile}
	
	docker build --pull \
	-t ${input_imagename} \
	-f ${dockerfile} .
	
	rm -rf ${content_dir}
	
	# tag and push the image to aws ECR
	docker tag ${input_imagename} ${fullname}
	docker push ${fullname}
}

input_imagename=
base_imagename="invivobase:latest"
branch="dev"
inputfile=
replace=
while getopts ':n:i:b:e:rh' OPTION
do
    case $OPTION in
        n)  fc "$OPTARG"
            input_imagename="$OPTARG";;
        i)  fc "$OPTARG"
        	inputfile="$OPTARG";;
        b)  fc "$OPTARG"
        	base_imagename="$OPTARG";;
        e)  fc "$OPTARG"
        	branch="$OPTARG";;
        r)  replace="--prune";;
        h)  usage;;
        \?) usage;;
        :)  echo "Missing option argument for -$OPTARG" >&2; exit 1 ;;

    esac
done
shift "$((OPTIND-1))"

if [ "" == "$input_imagename" ]; then
	printf " -n [option] is required. See help: [%s -h]\n" $(basename "$0") && exit 1
fi

build