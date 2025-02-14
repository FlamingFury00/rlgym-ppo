    def learn(self):
        """
        Function to wrap the _learn function in a try/catch/finally
        block to ensure safe execution and error handling.
        :return: None
        """
        try:
            print("Starting the learning process...")
            self._learn()
        except Exception as e:
            import traceback

            print("\\n\\nLEARNING LOOP ENCOUNTERED AN ERROR\\n")
            print(f"Error Type: {type(e).__name__}")
            print(f"Error Message: {e}")
            traceback.print_exc()

            try:
                print("Attempting to save progress before exiting...")
                self.save(self.agent.cumulative_timesteps)
            except Exception as save_error:
                print("FAILED TO SAVE ON EXIT")
                print(f"Save Error Type: {type(save_error).__name__}")
                print(f"Save Error Message: {save_error}")
                traceback.print_exc()
        finally:
            print("Performing cleanup...")
            self.cleanup()
            print("Cleanup completed. Exiting the learning process.")