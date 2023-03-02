import inquirer

# Define the list of questions to ask the user
questions = [
    inquirer.Text('name', message="What's your name?"),
    inquirer.List('gender', message="What's your gender?", choices=['Male', 'Female', 'Other']),
    inquirer.Checkbox('languages', message="What programming languages do you know?",
                      choices=['Python', 'Java', 'JavaScript', 'C++'])
]
