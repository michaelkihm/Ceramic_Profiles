FROM python:3.7

RUN apt-get install  -y git-core
RUN apt update -y && apt install build-essential -y &&  apt-get install manpages-dev -y
RUN pip install --upgrade pip
RUN pip install numpy && \
    pip install opencv-python && \
    pip install scikit-learn && \
    pip install pandas && \
    pip install pylint && \
    pip install jupyter && \
    pip install matplotlib && \
    pip install imutils

#add non root user to access locally data created in dev container 
ARG USERNAME=michael
RUN useradd -ms /bin/bash  $USERNAME -u 1000

# store vscode extensions installed in vscode devcontainer
RUN mkdir -p /home/$USERNAME/.vscode-server/extensions \
        /home/$USERNAME/.vscode-server-insiders/extensions \
    && chown -R $USERNAME \
        /home/$USERNAME/.vscode-server \
        /home/$USERNAME/.vscode-server-insiders
#store bash history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo $SNIPPET >> "/root/.bashrc" \
    # [Optional] If you have a non-root user
    && mkdir /commandhistory \
    && touch /commandhistory/.bash_history \
    && chown -R $USERNAME /commandhistory \
    && echo $SNIPPET >> "/home/$USERNAME/.bashrc" 
USER $USERNAME
ENTRYPOINT ["bash"]
#ENTRYPOINT ["jupyter", "notebook", "--ip='0.0.0.0'", "--no-browser", "--allow-root"]
