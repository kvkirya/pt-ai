IMAGE_URI := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(REPO_NAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

.PHONY: setup_repo build_image build_amd_image run_container debug_container push_container deploy_revision

setup_repo:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev
	gcloud artifacts repositories create $(REPO_NAME) --repository-format=docker \
		--location=$(GCP_REGION) --description="Repo for deploying my app"

build_image:
	docker build -t $(IMAGE_URI) .

build_amd_image:
	docker build -t $(IMAGE_URI) --platform=linux/amd64 .

run_container:
	docker run -it --rm -p 8080:8080 -e PORT=8080 $(IMAGE_URI)

debug_container:
	docker run -it --rm -p 8080:8080 -e PORT=8080 $(IMAGE_URI) /bin/bash

push_container:
	docker push $(IMAGE_URI)

deploy_revision:
	gcloud run deploy --image $(IMAGE_URI) --region $(GCP_REGION)

full_deploy: build_image push_container deploy_revision
