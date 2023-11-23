import pickle
from open_qstar.db.pgsql import ContextModel

class LearningContext:
    def __init__(self, instructions, parameters):
        self.instructions = instructions
        self.parameters = parameters

class ContextManager:
    def __init__(self, redis_client, sqlalchemy_session):
        self.contexts = {}
        self.current_context = None
        self.redis_client = redis_client
        self.sqlalchemy_session = sqlalchemy_session

    def switch_context(self, new_instruction_set):
        context_key = self.generate_context_key(new_instruction_set)
        if context_key in self.contexts:
            self.current_context = self.contexts[context_key]
        else:
            self.current_context = self.load_context_from_storage(context_key, new_instruction_set)
            self.contexts[context_key] = self.current_context

    def generate_context_key(self, instruction_set):
        return hash(tuple(instruction_set))

    def load_context_from_storage(self, context_key, instruction_set):
        cached_data = self.redis_client.get(context_key)
        if cached_data:
            return pickle.loads(cached_data)

        context_data = self.sqlalchemy_session.query(ContextModel).filter_by(key=context_key).first()
        if context_data:
            self.redis_client.set(context_key, context_data.data)
            return pickle.loads(context_data.data)

        return LearningContext(instruction_set, self.initialize_parameters())

    def save_context_to_storage(self, context_key, context_data):
        # Serialize and save the context data to PostgreSQL and Redis
        serialized_data = pickle.dumps(context_data)
        self.redis_client.set(context_key, serialized_data)

        db_context = ContextModel(key=context_key, data=serialized_data)
        self.sqlalchemy_session.add(db_context)
        self.sqlalchemy_session.commit()

    def initialize_parameters(self):
        # TODO Initialize model parameters for a new context
        # ...
        return parameters
