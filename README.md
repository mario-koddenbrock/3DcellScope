# 3DCellScope

3DCellScope is an advanced 3D image analysis platform using AI-driven segmentation to quantify morphological and topological changes in 3D cell cultures. It analyzes nuclear, cytoplasmic, and organoid-level features, providing insights into tissue patterning and cell organization, with applications in biomedical research and regenerative medicine.

## Installation Guide

### Prerequisites

Ensure that your Python version is **3.9.13**. The code may not work with Python 3.10 or higher without additional modifications, so please avoid using those versions.

### Setting Up the Environment

Unzip the folder containing the code and navigate to the project directory.

It is highly recommended to use a virtual environment to prevent conflicts if you're working on multiple Python projects. You can create and activate a virtual environment using the following steps:

### Creating a Virtual Environment

To create a virtual environment, run the following command:

`virtualenv name_of_your_venv`

If your default Python version is not 3.9, you can specify the correct version explicitly:

`virtualenv -p python3.9 name_of_your_venv`

### Activating the Virtual Environment

Once the virtual environment is created, activate it with the appropriate command:

On Windows:

`name_of_your_venv\Scripts\activate`

On macOS/Linux:

`source name_of_your_venv/bin/activate`

### Installing Dependencies

After activating the virtual environment, install the required dependencies by running:

`pip install -r requirements.txt`

The installation process might take a few minutes, but you should not encounter any warnings or errors.

## Running the Software

To launch the software, execute the following script:

`python OS_Demo.py`

Ensure that the virtual environment is activated before running the script.