#base image provides CUDA support on Ubuntu 16.04
FROM tableclassifier

#allow passing in a specific model to package
ARG MODEL


#drop the model in place
COPY ${MODEL} /model



#entrypoint used to serve with the preloaded model
WORKDIR /
EXPOSE 8888
ENTRYPOINT ["python", "-m", "tableclassifier", "serve", "/model"]