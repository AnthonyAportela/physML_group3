#!/bin/bash

# add all files to the Git repository
git add .

# prompt the user for a commit message
echo "Enter a commit message:"
read commit_message

# commit the changes with the custom message
git commit -m "$commit_message"

# push changes to the default branch of the GitHub repository
git push
