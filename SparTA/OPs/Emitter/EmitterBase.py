# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc


class EmitterBase(abc.ABC):

    @abc.abstractmethod
    def emit_function_call():
        """
        Emit the funtion call
        """

    @abc.abstractmethod
    def emit_function_body():
        """
        Emit the body of the function
        """

    @abc.abstractmethod
    def emit_dependency():
        """
        Emit the dependent headers
        """

    @abc.abstractmethod
    def emit_test_main():
        """
        Emit the main function used to test the speedup/memory footprint
        """ 
