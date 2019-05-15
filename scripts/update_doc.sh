#!/usr/bin/env bash
usage () { 
    echo
    printf "Usage: %s [OPTIONS] <ARGS>\n
    [OPTIONS] and corresponding <ARGS> are:\n
    [-r] <repository with the html doc to update> # Fall back to  bitbucket.org/invivoai/invivoai.bitbucket.io.git
    [-n] <where to write the current doc on invivoai.bitbucket.io > # default: Will use base by default for invivobase package
    [-a] <path> # add this path to the file to be pushed
    [-c] # clean all temporary files
    [-s] <strings for commit> default: Update documentation: current date
    [-d] <path to doc files> default: ../docs  (relative to this script)
    [-u] # upload documentation online
    [-h] # show this help screen then exit

    This script assume that you have GIT_ROBOT_USERNAME and GIT_ROBOT_PASSWD environment variable set.
    \n" "$0" 1>&2; exit 1; 
}

doc_update(){
    # get current path to this script
    cd $DIR # change this if not relative path
    pip install -r requirements.txt # ensure requirements are installed
    rm -rf build && make html && mkdir -p build/tmp
 
    # we need to set two environment variable in bitbucket pipeline
    # $GIT_ROBOT_USERNAME and $GIT_ROBOT_PASSWD for a bitbucket account, that has readonly
    # access to main repository but can write to the doc repository
    if  [ "$update" = true ] ; then
        cd build/tmp &&
        git clone "https://$GIT_ROBOT_USERNAME:$GIT_ROBOT_PASSWD@$DOC_REPO" . &&

        git rm -r $HTML_DIR;  mv ../html $HTML_DIR
        for l in  $(ls $HTML_DIR/*.html); do
            sed -i -e 's/<head>/<head>\n<meta name="robots" content="noindex" \/>/g' $l;  
        done

        if  [ "$addpath" = true ] ; then
            cp -r $DIR/$PATH_TO_ADD  .
        fi
        
        echo -e 'User-agent: *\nDisallow: /' > robots.txt
        git add -A . &&
        git commit -m "$COMMIT_STR [$HTML_DIR]" &&
        git push "https://$GIT_ROBOT_USERNAME:$GIT_ROBOT_PASSWD@$DOC_REPO" master
    fi

    if  [ "$clean_data" = true ] ; then
        echo "Cleaning build directory now ..."
        rm -rf "$DIR/build";
    fi
}

# set some defaults parameters
DOC_REPO="bitbucket.org/invivoai/invivoai.bitbucket.io.git" 
HTML_DIR="base"
COMMIT_STR="Update documentation: $(date "+%Y-%m-%d at %R")"
DIR="$( cd "$( dirname "$0" )" >/dev/null && pwd )/../docs"
clean_data=false
update=false
addpath=false
PATH_TO_ADD="docs/coverage"
# parse the options from the command line
while getopts ':r:d:s:a:hcun:' OPTION
do
    case $OPTION in
        r)  fc "$OPTARG"
            DOC_REPO="$OPTARG"
            ;;
        c)  clean_data=true
            ;;
        u)  update=true
            ;;
        n)  fc "$OPTARG"
            HTML_DIR="$OPTARG"
            ;;
        a)  fc "$OPTARG"
            PATH_TO_ADD="$OPTARG"
            addpath=true
            ;;
        s)  COMMIT_STR="$OPTARG"
            ;;
        d)  fc "$OPTARG"
            DIR="$OPTARG"
            if [ ! -d "$DIR" ]; then
                echo "Folder $DIR provided to option -d does not exist"
                exit 1
            fi
            ;;
        h)  usage;;
        \?)  echo 
            echo "Invalid option: -$OPTARG" >&2
            usage;;
        :)  echo 
            echo "Option -$OPTARG requires an argument." >&2
            usage;;
    esac
done
shift $((OPTIND - 1))

doc_update
