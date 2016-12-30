---
layout: post
title: "Jekyll+github Personal Blog"
date: 2016-12-29  
description: "Use Jekyll+github to deploy personal blog"  

tag: TOOL 
---   
    Let's deploy your personal blog with Jekyll+github.

## CONTENT：
+   Step0: install Ruby  
    1 install RVM  
    $curl -L https://get.rvm.io | bash -s stable  
    $source ~/.rvm/scripts/rvm  
    $rvm -v #check the version  
     
    2 install ruby  
    $sudo rvm list known    # list all the ready ruby  
    $ruby -v                #old version ruby 2.0.0p648 (2015-12-16 revision 53162) [universal.x86_64-darwin15]
    $sudo rvm install 2.0.0

    3 if you live in China, you might not be able to access to rubygems.
    $gem source -r https://rubygems.org/  
    $gem sources -a https://gems.ruby-china.org/  
    $gem sources -l

+   Step1: install gem    
    $gem -v     #check the version  
    $sudo gem update --system #update gem  

+   Step2: install jekyll   
    $sudo gem install jekyll   
    $jekyll -v # check the version  


+   Step3:  
    $jekyll build  
    $jekyll server
    then, vist through browser: http://127.0.0.1:4000  
    or  
    $cd git_workspace/guihongwan.github.io  
    $bundle exec jekyll build  
    $bundle exec jekyll server  
    Use your browser to visit:http://127.0.0.1:4000/

+   tips:  
1.  update jekyll:  
    $gem update jekyll
2.  reference:  
    https://help.github.com/categories/github-pages-basics/  
    http://baixin.io/2016/10/jekyll_tutorials1/
3.  problems:  
    p1: Operation not permitted - /usr/bin/kramdown  
        solution: sudo gem install jekyll --user-install  

    p2: Error installing jekyll:  
         ERROR: Failed to build gem native extension.  
        solution: please install “xcode 6 command line tools” first. $ xcode-select --install  

    p3: Operation not permitted - /usr/bin/listen  
        solution:    
            $/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"  #brew, also named brew, is as same as apt-get in Ubuntu http://brew.sh/     
            $brew install ruby  #brew uninstall ruby

