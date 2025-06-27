import logfire

logfire.configure(environment='hello-world')

logfire.info('hello {place}', place='world')
