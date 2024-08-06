import logging

class PrintLogo:
    def __init__(self):
        """
        Initializes the PrintLogo class and logs the ASCII art logo.
        """
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)

        # Ensure that the logging handler is set up only once
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.log.addHandler(handler)

        self.log.info(self.get_logo())

    @staticmethod
    def get_logo():
        """
        Returns the ASCII art logo as a string.
        """
        return """
                _____      _              _                                            _      __      ________ 
               / ____|    | |            | |                                          | |     \ \    / /____  |
              | |    _   _| |_ ___  _ __ | |__   ___ _ __   ___   __ _ _ __ __ _ _ __ | |__    \ \  / /    / / 
              | |   | | | | __/ _ \| '_ \| '_ \ / _ \ '_ \ / _ \ / _` | '__/ _` | '_ \| '_ \    \ \/ /    / /  
              | |___| |_| | || (_) | |_) | | | |  __/ | | | (_) | (_| | | | (_| | |_) | | | |    \  /    / /   
               \_____\__, |\__\___/| .__/|_| |_|\___|_| |_|\___/ \__, |_|  \__,_| .__/|_| |_|     \(_)  /_/    
                      __/ |        | |                            __/ |         | |                            
                     |___/         |_|                           |___/          |_|                            
        """
