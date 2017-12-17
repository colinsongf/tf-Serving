import os
import sys
import tensorflow as tf
def export(model, sess, signature_name, export_path, version):
    # export path
    export_path = os.path.join(export_path, signature_name, str(version))
    print('Exporting trained model to {} ...'.format(export_path))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    # Build the signature_def_map.
    input_data1= tf.saved_model.utils.build_tensor_info(model.input_data1)
    input_data2= tf.saved_model.utils.build_tensor_info(model.input_data2)
    input_data3= tf.saved_model.utils.build_tensor_info(model.input_data3)
    input_data4= tf.saved_model.utils.build_tensor_info(model.input_data4)
    output=tf.saved_model.utils.build_tensor_info(model.output1)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'input_data1': input_data1, 'input_data2': input_data2, 'input_data3': input_data3, 'input_data4': input_data4 },
        outputs={'output': output},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)  # 'tensorflow/serving/predict'
    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            signature_name: prediction_signature
        })
    builder.save()
if __name__=='__main__':
    pass
