docker run -it --net=host \
            --ipc=host \
            -v $(pwd):/server \
            --name DongLaoAI \
            hopny bash