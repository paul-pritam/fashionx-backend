import 'dart:io';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

class RecommendationFromImages extends StatefulWidget {
  const RecommendationFromImages({Key? key}) : super(key: key);

  @override
  State<RecommendationFromImages> createState() =>
      _RecommendationFromImagesState();
}

class _RecommendationFromImagesState extends State<RecommendationFromImages> {
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
        title: const Text('Recommendation from images'),
      ),
      body: Center(
        child: Row(
          children: [
            if (_image != null)
              Container(
                width: double.infinity,
                height: 500,
                child: Image.file(_image!),
              ),
            SizedBox(
              width: 200,
              child: ElevatedButton.icon(
                icon: const Icon(Icons.camera_alt_outlined),
                onPressed: () => _pickImage(ImageSource.camera,),
                label: const Text('Open Camera'),
              ),
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
