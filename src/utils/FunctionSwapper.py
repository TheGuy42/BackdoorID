import types


class FunctionSwapper:
    def __init__(self, obj, original_func_name, replacement_func, takes_self=False):
        """
        obj: The object containing the method to replace
        original_func_name: Name of the method to temporarily replace (as string)
        replacement_func: The function to use as replacement
        takes_self: Whether the replacement function takes self as first argument (default: True)
        """
        self.obj = obj
        self.original_func_name = original_func_name
        self.replacement_func = replacement_func
        self.takes_self = takes_self
        self.original_func = None
    
    def __enter__(self):
        # Store the original function
        self.original_func = getattr(self.obj, self.original_func_name)
        # Replace with the new function
        if self.takes_self:
            # Bind the function to the object so it receives self
            bound_method = types.MethodType(self.replacement_func, self.obj)
            setattr(self.obj, self.original_func_name, bound_method)
        else:
            # Wrap it to ignore self and bind the wrapper
            # def wrapper(self_arg, *args, **kwargs):
            #     return self.replacement_func(*args, **kwargs)
            # bound_wrapper = types.MethodType(wrapper, self.obj)
            # setattr(self.obj, self.original_func_name, bound_wrapper)
            setattr(self.obj, self.original_func_name, self.replacement_func)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original function
        setattr(self.obj, self.original_func_name, self.original_func)


class VarSwapper:
    def __init__(self, *swaps):
        """
        swaps: Variable number of tuples, each containing (obj, var_name, replacement_value)
               - obj: The object containing the variable to replace
               - var_name: Name of the variable to temporarily replace (as string)
               - replacement_value: The value to use as replacement
        """
        self.swaps = swaps
        self.original_values = []
    
    def __enter__(self):
        # Store original values and perform swaps
        for obj, var_name, replacement_value in self.swaps:
            original_info = {
                'obj': obj,
                'var_name': var_name,
                'had_original': hasattr(obj, var_name),
                'original_value': getattr(obj, var_name) if hasattr(obj, var_name) else None
            }
            self.original_values.append(original_info)
            # Replace with the new value
            setattr(obj, var_name, replacement_value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original values in reverse order
        for original_info in reversed(self.original_values):
            if original_info['had_original']:
                setattr(original_info['obj'], original_info['var_name'], original_info['original_value'])
            else:
                # If the variable didn't exist before, delete it
                delattr(original_info['obj'], original_info['var_name'])


if __name__ == "__main__":

    class SampleClass:
        def method_to_replace(self):
            return "Original Method"
        
    def new_method_with_self(self):
        return "Replaced Method (with self)"
    
    def new_method_without_self():
        return "Replaced Method (without self)"
    
    sample = SampleClass()
    print(sample.method_to_replace())  # Should print "Original Method"
    
    # Example 1: Replacement function that takes self
    with FunctionSwapper(sample, 'method_to_replace', new_method_with_self, takes_self=True):
        print(sample.method_to_replace())  # Should print "Replaced Method (with self)"
    
    # Example 2: Replacement function that doesn't take self
    with FunctionSwapper(sample, 'method_to_replace', new_method_without_self, takes_self=False):
        print(sample.method_to_replace())  # Should print "Replaced Method (without self)"
    
    print(sample.method_to_replace())  # Should print "Original Method" again
    
    print("\n--- Testing VarSwapper ---")
    
    class SampleClassWithVars:
        def __init__(self):
            self.my_var = "Original Value"
            self.another_var = 42
    
    sample_vars = SampleClassWithVars()
    print(f"my_var: {sample_vars.my_var}")  # Should print "Original Value"
    print(f"another_var: {sample_vars.another_var}")  # Should print 42
    
    # Example 1: Replace a single variable (backward compatible)
    with VarSwapper((sample_vars, 'my_var', 'Replaced Value')):
        print(f"my_var (inside context): {sample_vars.my_var}")  # Should print "Replaced Value"
    
    print(f"my_var (after context): {sample_vars.my_var}")  # Should print "Original Value" again
    
    # Example 2: Replace multiple variables at once
    with VarSwapper(
        (sample_vars, 'my_var', 'New Value'),
        (sample_vars, 'another_var', 999)
    ):
        print(f"my_var (inside context): {sample_vars.my_var}")  # Should print "New Value"
        print(f"another_var (inside context): {sample_vars.another_var}")  # Should print 999
    
    print(f"my_var (after context): {sample_vars.my_var}")  # Should print "Original Value"
    print(f"another_var (after context): {sample_vars.another_var}")  # Should print 42
    
    # Example 3: Add a new variable temporarily
    print(f"Has new_var before: {hasattr(sample_vars, 'new_var')}")  # Should print False
    with VarSwapper((sample_vars, 'new_var', 'Temporary Variable')):
        print(f"new_var (inside context): {sample_vars.new_var}")  # Should print "Temporary Variable"
    print(f"Has new_var after: {hasattr(sample_vars, 'new_var')}")  # Should print False
    
    # Example 4: Multiple objects
    sample_vars2 = SampleClassWithVars()
    with VarSwapper(
        (sample_vars, 'my_var', 'Object 1 Changed'),
        (sample_vars2, 'my_var', 'Object 2 Changed')
    ):
        print(f"sample_vars.my_var: {sample_vars.my_var}")  # Should print "Object 1 Changed"
        print(f"sample_vars2.my_var: {sample_vars2.my_var}")  # Should print "Object 2 Changed"
    
    print(f"sample_vars.my_var (restored): {sample_vars.my_var}")  # Should print "Original Value"
    print(f"sample_vars2.my_var (restored): {sample_vars2.my_var}")  # Should print "Original Value"