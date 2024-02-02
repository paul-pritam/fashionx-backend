import 'package:flutter/material.dart';
import 'package:try2/pages/login.dart';
import '../pages/recommendation from image.dart';
import '../pages/recommendationfromsketches.dart';

class MyPage extends StatefulWidget {
  const MyPage({Key? key}) : super(key: key);

  @override
  State<MyPage> createState() => _MyPageState();
}

class _MyPageState extends State<MyPage> {
  final _categories = [
    {
      'name': 'Shoes',
      'image': 'lib/assets/dataset_temp/25850.png',
      'dataset_temp': [
        {'productName': 'Product X1', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X2', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X3', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X4', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X5', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X6', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X7', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X8', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X9', 'productImage': 'lib/assets/dataset_temp/x.png'},
        {'productName': 'Product X10', 'productImage': 'lib/assets/dataset_temp/x.png'},
      ],
    },
    {
      'name': 'y',
      'image': 'lib/assets/dataset_temp/y.png',
      'dataset_temp': [
        {'productName': 'Product Y1', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y2', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y3', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y4', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y5', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y6', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y7', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y8', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y9', 'productImage': 'lib/assets/dataset_temp/y.png'},
        {'productName': 'Product Y10', 'productImage': 'lib/assets/dataset_temp/y.png'},
      ],
    },
    {
      'name': 'z',
      'image': 'lib/assets/dataset_temp/z.png',
      'dataset_temp': [
        {'productName': 'Product Z1', 'productImage': 'lib/assets/dataset_temp/z.png'},
        {'productName': 'Product Z2', 'productImage': 'lib/assets/dataset_temp/z.png'},
        {'productName': 'Product Z3', 'productImage': 'lib/assets/dataset_temp/z.png'},
        {'productName': 'Product Z4', 'productImage': 'lib/assets/dataset_temp/z.png'},
        {'productName': 'Product Z5', 'productImage': 'lib/assets/dataset_temp/z.png'},
        {'productName': 'Product Z6', 'productImage': 'lib/assets/dataset_temp/z6.png'},
        {'productName': 'Product Z7', 'productImage': 'lib/assets/dataset_temp/z7.png'},
        {'productName': 'Product Z8', 'productImage': 'lib/assets/dataset_temp/z8.png'},
        {'productName': 'Product Z9', 'productImage': 'lib/assets/dataset_temp/z9.png'},
        {'productName': 'Product Z10', 'productImage': 'lib/assets/dataset_temp/z10.png'},
      ],
    },
    // {
    //   'name': 'z',
    //   'image': 'lib/assets/images/z.png',
    //   'dataset_temp': [
    //     {'productName': 'Product Z1', 'productImage': 'lib/assets/dataset_temp/z1.png'},
    //     {'productName': 'Product Z2', 'productImage': 'lib/assets/dataset_temp/z2.png'},
    //     {'productName': 'Product Z3', 'productImage': 'lib/assets/dataset_temp/z3.png'},
    //     {'productName': 'Product Z4', 'productImage': 'lib/assets/dataset_temp/z4.png'},
    //     {'productName': 'Product Z5', 'productImage': 'lib/assets/dataset_temp/z5.png'},
    //     {'productName': 'Product Z6', 'productImage': 'lib/assets/dataset_temp/z6.png'},
    //     {'productName': 'Product Z7', 'productImage': 'lib/assets/dataset_temp/z7.png'},
    //     {'productName': 'Product Z8', 'productImage': 'lib/assets/dataset_temp/z8.png'},
    //     {'productName': 'Product Z9', 'productImage': 'lib/assets/dataset_temp/z9.png'},
    //     {'productName': 'Product Z10', 'productImage': 'lib/assets/dataset_temp/z10.png'},
    //   ],
    // },
    // {
    //   'name': 'z',
    //   'image': 'lib/assets/images/z.png',
    //   'dataset_temp': [
    //     {'productName': 'Product Z1', 'productImage': 'lib/assets/dataset_temp/z1.png'},
    //     {'productName': 'Product Z2', 'productImage': 'lib/assets/dataset_temp/z2.png'},
    //     {'productName': 'Product Z3', 'productImage': 'lib/assets/dataset_temp/z3.png'},
    //     {'productName': 'Product Z4', 'productImage': 'lib/assets/dataset_temp/z4.png'},
    //     {'productName': 'Product Z5', 'productImage': 'lib/assets/dataset_temp/z5.png'},
    //     {'productName': 'Product Z6', 'productImage': 'lib/assets/dataset_temp/z6.png'},
    //     {'productName': 'Product Z7', 'productImage': 'lib/assets/dataset_temp/z7.png'},
    //     {'productName': 'Product Z8', 'productImage': 'lib/assets/dataset_temp/z8.png'},
    //     {'productName': 'Product Z9', 'productImage': 'lib/assets/dataset_temp/z9.png'},
    //     {'productName': 'Product Z10', 'productImage': 'lib/assets/dataset_temp/z10.png'},
    //   ],
    // },
    // {
    //   'name': 'z',
    //   'image': 'lib/assets/images/z.png',
    //   'dataset_temp': [
    //     {'productName': 'Product Z1', 'productImage': 'lib/assets/dataset_temp/z1.png'},
    //     {'productName': 'Product Z2', 'productImage': 'lib/assets/dataset_temp/z2.png'},
    //     {'productName': 'Product Z3', 'productImage': 'lib/assets/dataset_temp/z3.png'},
    //     {'productName': 'Product Z4', 'productImage': 'lib/assets/dataset_temp/z4.png'},
    //     {'productName': 'Product Z5', 'productImage': 'lib/assets/dataset_temp/z5.png'},
    //     {'productName': 'Product Z6', 'productImage': 'lib/assets/dataset_temp/z6.png'},
    //     {'productName': 'Product Z7', 'productImage': 'lib/assets/dataset_temp/z7.png'},
    //     {'productName': 'Product Z8', 'productImage': 'lib/assets/dataset_temp/z8.png'},
    //     {'productName': 'Product Z9', 'productImage': 'lib/assets/dataset_temp/z9.png'},
    //     {'productName': 'Product Z10', 'productImage': 'lib/assets/dataset_temp/z10.png'},
    //   ],
    // },


  ];

  int _currentCategoryIndex = 0;
  late PageController _pageController;

  @override
  void initState() {
    super.initState();
    _pageController = PageController(
      initialPage: _currentCategoryIndex,
      viewportFraction: 0.3, // Adjust the fraction as needed
    );
  }
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.menu, color: Colors.black),
          onPressed: () => _scaffoldKey.currentState?.openDrawer(),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications_active, color: Colors.black),
            onPressed: () {
              // Handle notifications
            },
          ),
        ],
      ),
      drawer: Drawer(
        child: Container(
          color: Colors.blue,
          child: ListView(
            padding: EdgeInsets.zero,
            children: [
              DrawerHeader(
                decoration: BoxDecoration(
                  color: Colors.blue,
                ),
                child: Text(
                  'FashionX',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 50,
                  ),
                ),
              ),
              ListTile(
                title: const Text(
                  'Recommendation from Images',
                  style: TextStyle(color: Colors.white),
                ),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const RecommendationFromImages(),
                    ),
                  );
                },
              ),
              ListTile(
                title: const Text(
                  'Recommendation from Sketch',
                  style: TextStyle(color: Colors.white),
                ),
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const RecommendationFromSketches(),
                    ),
                  );
                },
              ),
              ListTile(
                title: const Text(
                  'login',
                  style: TextStyle(color: Colors.white),
                ),
                onTap: () {

                },
              ),
            ],
          ),
        ),
      ),
      body: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          SizedBox(
            height: 150,
            child: PageView.builder(
              controller: _pageController,
              itemCount: _categories.length,
              onPageChanged: (index) {
                setState(() => _currentCategoryIndex = index);
              },
              itemBuilder: (context, index) {
                return Container(
                  margin: const EdgeInsets.symmetric(horizontal: 10),
                  child: Column(
                    children: [
                      CircleAvatar(
                        radius: 50,
                        backgroundImage: AssetImage(_categories[index]['image'] as String? ?? ''),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        _categories[index]['name']?.toString() ?? 'Default Text',
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
          const SizedBox(height: 20),
          Expanded(
            child: GridView.builder(
              gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                crossAxisCount: 2,
                crossAxisSpacing: 8.0,
                mainAxisSpacing: 8.0,
              ),
              itemCount: (_categories[_currentCategoryIndex]['dataset_temp'] as List<Map<String, dynamic>>?)?.length ?? 0,
              itemBuilder: (context, index) {
                final List<Map<String, dynamic>>? dataset_temp =
                _categories[_currentCategoryIndex]['dataset_temp'] as List<Map<String, dynamic>>?;
                final Map<String, dynamic>? product = dataset_temp?.elementAt(index);

                if (product != null) {
                  return Card(
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(10.0),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        ClipRRect(
                          borderRadius: BorderRadius.circular(10.0),
                          child: Image.asset(
                            product['productImage']?.toString() ?? '',
                            height: 120,
                            width: double.infinity,
                            fit: BoxFit.cover,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 8.0),
                          child: Text(
                            product['productName']?.toString() ?? '',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                } else {
                  return Container();
                }
              },
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }
}
