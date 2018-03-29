from keras.models import load_model

model = load_model('first_try.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

score = model.evaluate(test_data, test_label, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
