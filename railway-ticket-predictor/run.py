from app import create_app

# This line imports the function that builds your Flask app
# from your 'app' folder.
app = create_app()

# This is a standard Python check to ensure the code inside this block
# only runs when you execute this file directly.
if __name__ == '__main__':
    
    # This command starts the web server.
    # 'debug=True' is a useful setting for development that automatically
    # reloads the server when you make code changes.
    app.run(debug=True)
