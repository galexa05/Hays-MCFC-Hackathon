# Creating a Pipenv Environment

To create a new Pipenv environment, follow these steps:

- Open a terminal window and navigate to your project directory.

- Make sure you have Pipenv installed. If you don't have it installed, you can install it using pip by typing the following command:


```sh
pip install pipenv
```

- Once you have Pipenv installed, create a new environment by typing the following command:
```sh
pipenv --python 3.9
```
This command will create a new environment using Python 3.9. If you want to use a different version of Python, replace 3.9 with the version you want to use.

- Pipenv will create a new environment and a Pipfile in your project directory. The Pipfile is a configuration file that contains a list of the packages required by your project.

- Install the packages required by your project by typing the following command:
```sh
pipenv install
```
This command will install the packages listed in the Pipfile.

- Activate the environment by typing the following command:
```sh
pipenv shell
```
This command will activate the environment and change your shell prompt to indicate that you are working inside the environment.


You can install new packages by typing pipenv install <package-name> and they will be added to the Pipfile. You can also run Python scripts and commands inside the environment by typing python <script-name> or python -c "<command>".

ou can install new packages by typing 
```sh
pipenv install "<package-name>"
```

and they will be added to the Pipfile. You can also run Python scripts and commands inside the environment by typing 

```sh
python "<script-name>" 
```

or 

```sh
python -c "<command>"
``` 
