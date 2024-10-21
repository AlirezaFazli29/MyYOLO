# Instructions

To get started with this project, please follow these steps:

1. Clone the repository:
   Use the following command to download the project:
   ```bash
   git pull <repository-url>

2. Create a virtual environment: Run the command below to create a virtual environment:
   ```bash
   python -m venv venv

3. Activate the virtual environment:
   On windows:
   ```shell
   venv\Scripts\activate
   ```
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   
4. Install the required packages: Make sure you have all the dependencies by running:
   ```bash
   pip install -r requirements.txt

5. Run the application: Finally, start the application with the following command:
   ```bash
   uvicorn app:app --reload

Now you should be able to access the application in your browser!
