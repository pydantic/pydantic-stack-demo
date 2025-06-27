import logfire

logfire.configure(service_name='hello-world')

logfire.info('hello {place}', place='world')
