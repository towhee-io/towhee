import inspect
import marshal
import types
import dill

#use case
def f(callable, input):
    return map(callable, input)

#closure fail lambda
def closure_lambda():
    bar = 1
    da_lamba = lambda x: x + bar
    return da_lamba

#clean lambda
def regular_lambda():
    da_lamba = lambda x: x + 1
    return da_lamba

#String
def s(callable):
    return str(inspect.getsourcelines(callable)[0])

#Marshall
def m(callable, input):
    callable_serialized = marshal.dumps(callable.__code__)
    new_code = marshal.loads(callable_serialized)
    new = types.FunctionType(new_code, globals())
    return map(new, input)

#Dill
def d(callable, input):
    callable_serialized = dill.dumps(callable)
    new = dill.loads(callable_serialized)
    return map(new, input)


test = [1, 2, 3]


marshal_good_lambda = m(regular_lambda(), test)
print('marshal_good_lambda:', list(marshal_good_lambda))
try:
    marshal_bad_lambda = m(closure_lambda(), test)
except Exception as e:
    print('marshal_bad_lambda:', e)

dill_good_lambda = d(regular_lambda(), test)
print('dill_good_lambda:', list(dill_good_lambda))

dill_bad_lambda = d(closure_lambda(), test)
print('dill_bad_lambda:', list(dill_bad_lambda))

print(s(regular_lambda()))
print(s(lambda x: x + 1))


