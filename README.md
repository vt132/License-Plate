# License Plate recognition

This is the advanced project for Advanced AI programming - , by Tran Tan Loc - 24C15011 and Vu Thien Hoang - 24C15034
## Description
This app provide 2 (+1 authentication) APIs and admin UIs to recognize and manage license plate using Django, FastAPI with 2 YOLOv5 models running in the background (one for plate detection and one for character detection) along with optional upscaler using community-finetuned ESRGAN.
- All APIs requires JWT authentication to operate, which provided by using the login API. First it use Django ORM query to find the user with the username, after that the password is being verified with hashed password in the database (hash with pdfbk2-sha256 by default). After that, a JWT token hashed with HS256 algorithmn (that include user UUID as data) will be return as a response, along with expiration time.
- After being authenticated with JWT, user can access to 2 APIs for license plate management: create and read license plate. In creating license plate data, user can input the number of the plate and if that plate is a wanted plate (in the meaning of criminal plate). When reading the license plate, user can input an image for the system to detect all license plate in that image and it will return every license plate number registered in the database. It also return the wanted value for the UI to handle the suitable response for it.
- This app is also come with a simple Django UI to manage Users and License Plate with create, update, delete actions built-in.

## How to run
- Install python (3.9+ recommended) and [pytorch](https://pytorch.org/get-started/locally/) (recommended to install the right version for your system)
- Run `git clone https://github.com/vt132/yolov5.git` to clone yolov5 repo for running models. Also download pretrained models for APIs to works.
- Run `pip install -r requirements.txt` to install required packages
- Run `python manage.py migrate` to create local database
- Run `python manage.py createsuperuser` to create admin user for Admin page access (/app/admin/local)
- Download 2 pretrained YOLO models and 1 fine-tuned ESRGAN model (from NMKD) (link at the resources section)
- Run `uvicorn main:app` to boot up the backend service
- Run `streamlit ui.py` to run demo UI (add `----server.enableXsrfProtection false` if encounter crsf error while running)
  
## Resources
Models: 
- [Plate detection](https://drive.google.com/file/d/1wnbyyVZzZJoMwhBYFdYQ578648mKMJFX/view?usp=sharing)
- [Character detection](https://drive.google.com/file/d/1WeOq1l41-3cV7tgDNVPxl03k_52atGPb/view?usp=sharing)
- [Upscaler](https://drive.google.com/file/d/1b61PEktul5tzkLQsFRATjAZwvla_KLS_/view?usp=drive_link)
Old Demo: [Link](https://youtu.be/0H4fDBCdg2g)
