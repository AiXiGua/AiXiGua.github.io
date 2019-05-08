---
author: linhu
title: "Anaconda environment transfer --- Anaconda 环境多服务器间迁移"
comments: true
classes: wide
date: 2019-05-09 17:42:32+00:00
categories:
  - Experience Share
tags:
  - Anaconda
---

## 核心命令
> conda env export > environment.yaml
> conda env create -f environment.yaml

## 技巧
将项目的环境搭建好之后，导出环境配置，并随着该项目一起使用git管理，当需要迁移到其他服务器时，随着git clone一起转移过去，并使用conda命令创建环境，简单快捷可维护。
