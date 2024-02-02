
import 'dart:async';
import 'dart:collection';

import 'package:flutter/material.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:try2/pages/login.dart';
import 'package:try2/prac/Q.dart';



void main() async {
  await Supabase.initialize(url: "https://inuqmekawvnmspgdqqmu.supabase.co", anonKey: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImludXFtZWthd3ZubXNwZ2RxcW11Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTkyMTQ4NDYsImV4cCI6MjAxNDc5MDg0Nn0.sxmb6_X9XQ7wPhOuLuF9WJjj_GyqAVrqETmEhLfXX-o",);
  runApp(const MyApp());
}
final supabase = Supabase.instance.client;
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(

        debugShowCheckedModeBanner: false,
        home:MyPage()
    );
  }
}
