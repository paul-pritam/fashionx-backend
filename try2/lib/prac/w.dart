import 'package:flutter/material.dart';

class EcommerceInterface extends StatelessWidget {
  const EcommerceInterface({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ecommerce Interface'),
      ),
      body: ListView.builder(
        itemCount: products.length,
        itemBuilder: (context, index) {
          Product product = products[index];
          return Card(
            child: ListTile(
              leading: Image.asset(product.imageUrl),
              title: Text(product.name),
              subtitle: Text(product.description),
              trailing: Text('\$${product.price}'),
            ),
          );
        },
      ),
    );
  }
}

class Product {
  final String imageUrl;
  final String name;
  final String description;
  final double price;

  Product({
    required this.imageUrl,
    required this.name,
    required this.description,
    required this.price,
  });
}

List<Product> products = [
  Product(
    imageUrl: 'assets/images/product1.jpg',
    name: 'Product 1',
    description: 'Description of Product 1',
    price: 100.00,
  ),
  Product(
    imageUrl: 'assets/images/product2.jpg',
    name: 'Product 2',
    description: 'Description of Product 2',
    price: 200.00,
  ),
  Product(
    imageUrl: 'assets/images/product3.jpg',
    name: 'Product 3',
    description: 'Description of Product 3',
    price: 300.00,
  ),
  Product(
    imageUrl: 'assets/images/product4.jpg',
    name: 'Product 4',
    description: 'Description of Product 4',
    price: 400.00,
  ),
  Product(
    imageUrl: 'assets/images/product5.jpg',
    name: 'Product 5',
    description: 'Description of Product 5',
    price: 500.00,
  ),
];
