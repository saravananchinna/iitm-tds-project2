# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.13

# Copy your function code into the container
COPY app.py ${LAMBDA_TASK_ROOT}

# (Optional) Copy any requirements.txt and install packages
COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to your handler (file.function)
CMD ["app.handler"]