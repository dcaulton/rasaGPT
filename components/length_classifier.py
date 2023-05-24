from typing import Dict, Text, Any, List, Type

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer

# TODO: Correctly register your component with its type
@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER], is_trainable=True
)


class LengthClassifier(GraphComponent):

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [Tokenizer]
    
    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls()
   

    def train(self, training_data: TrainingData) -> Resource:
        for example in training_data.training_examples:
            tokens = self.tokenizer.tokenize(example.text)
            length = len(tokens)
            example.set("length", length)



    def process(self, messages: List[Message]) -> List[Message]:
        # This method is used to modify the user message and remove the () if they are included in the user test.
        for message in messages:
            if 'text' in message.data.keys():
                tokens = Tokenizer.tokenize(message.data['text'])
                length = len(tokens)
                message.set("length", length)
        return messages