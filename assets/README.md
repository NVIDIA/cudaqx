# Git LFS Setup and Usage Guide

## Installation

Git LFS must be installed before using it in a repository. Follow the installation steps based on your operating system.

### Ubuntu (Debian-based distributions)
```sh
sudo apt update
sudo apt install git-lfs
git lfs install

### AlmaLinux and ManyLinux
sudo dnf install git-lfs
git lfs install

### macOS
brew install git-lfs
git lfs install
```

## Tracking and committing large files

1. Initialize Git LFS in your repository: 

    `git lfs install` 

2. Track specific file types or individual files using the following command: 

    `git lfs track "assets/*"`, where `assets` is a directory containing large files.

3. Commit the changes to `.gitattributes`: 

    `git add .gitattributes && git commit -m "Track large files with Git LFS"`

4. Add and commit the large files: 

    `git add assets/largefile.zip && git commit -m "Add large file"`

5. Push to remote:

    `git push origin branch_name`

## Cloning and fetching large files

1. Clone a repository that uses Git LFS:

    `git clone https://github.com/username/repository.git`. By default, cloning only retrieves the pointer files to the large file. To fetch the actual large files, use `git lfs pull`.

2. Fetch large files for an existing repository:

    `git lfs pull`

## Check Git LFS status

To check which files are tracked by Git LFS:

    `git lfs ls-files` 

## Removing a file from LFS

Use the following steps to remove a file from LFS:

    `git rm --cached assets/largefile.zip`, then commit and push.

    Once the file is removed, remember to delete the tracking information in `.gitattributes`.

