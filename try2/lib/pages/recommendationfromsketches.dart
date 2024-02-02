import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../prac/imagehelper.dart';

class RecommendationFromSketches extends StatefulWidget {
  const RecommendationFromSketches({Key? key}) : super(key: key);

  @override
  State<RecommendationFromSketches> createState() =>
      _RecommendationFromSketchesState();
}

class _RecommendationFromSketchesState extends State<RecommendationFromSketches> {
  File? _image;

  Future<void> _pickImage(ImageSource source) async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: source);
    if (image != null) {
      setState(() {
        _image = File(image.path);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Recommendation from sketches'),
      ),
      body: Center(
        child: Column(

          children: [
            if (_image != null)
              SizedBox(
                width: 500,
                height: 500,
                child: Image.file(_image!),
              ),
            SizedBox(
              // ignore: sort_child_properties_last
              child: ElevatedButton.icon(
                icon: const Icon(Icons.camera_alt_outlined),
                onPressed:() => _pickImage(ImageSource.camera,),
                label: const Text('Open Camera'),
              ),

              width: 200,
            ),
            SizedBox(
              width: 200,
              child: ElevatedButton.icon(
                icon: const Icon(Icons.image_outlined),
                onPressed: () => _pickImage(ImageSource.gallery),
                label: const Text('Open Gallery'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

