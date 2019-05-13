#!/bin/bash

## This script downloads stock options data, 
## veryfies zipfile integrity, saves md5 signature 
## and uploads them to S3.
## To run, pass a list of files to download:
## $> ./backup.sh files.txt

TMPDIR=tmp
NOW=$(date +"%m-%d-%Y-%H%M%S")
RETRY="${NOW}.txt"
MD5SUMS=md5sums.txt

mkdir -p $TMPDIR

while read filename
do
	echo "Downloading file $filename to $TMPDIR"
	wget --quiet -P $TMPDIR "ftp://l3_hdall:JKNRH7LYXV@ftp.deltaneutral.com/${filename}"

	newpath="$TMPDIR/$filename"
	echo "Verifying zipfile $newpath"
	if unzip -t -q $newpath
	then
		echo "File check OK"
	else
		echo "ERROR: File check failed for $newfile"
		echo $filename >> $RETRY
		rm $newpath
		continue
	fi

	echo "Appending md5 sum for $f"
	md5sum $newpath >> $MD5SUMS
    
	echo "Copying $newpath to S3 bucket"
	rclone copy -v $newpath longueduree:longueduree
	
	echo "Deleting $newpath"
	rm $newpath
done <$1
