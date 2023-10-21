#! /bin/bash

# cd scratch place
if [ -d ./example_data ]
then 
    echo "Dir example_data already exists."
else
    mkdir example_data/
fi
cd example_data/

# Download zip dataset from Google Drive
filename0='img_data.zip'
fileid0='17fiub4POzxQMZho5qnAFkE97dlH8kTRf'
filename1='pth_data.zip'
fileid1='1ji-SEZ8xusSmX4OmpPGVcMI4o27PEaL2'
if [ -d ./img ]
then 
    echo "Dir img already exists."
else
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid0}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid0}" -O ${filename0} && rm -rf /tmp/cookies.txt
    unzip -q ${filename0}
    # rm ${filename0}
fi

if [ -d ./pth ]
then 
    echo "Dir pth already exists."
else
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid1}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid1}" -O ${filename1} && rm -rf /tmp/cookies.txt
    unzip -q ${filename1}
    # rm ${filename1}
fi

