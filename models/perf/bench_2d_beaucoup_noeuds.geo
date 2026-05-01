SetFactory("OpenCASCADE");
lc = 0.035;
L = 2.0;
H = 1.4;
th = 0.07;

Rectangle(1) = {0, 0, 0, L, H};
Rectangle(2) = {0.15, 0.62, 0, 1.35, th};
Rectangle(3) = {0.82, 0.12, 0, th, 0.46};
Rectangle(4) = {0.82, 0.84, 0, th, 0.44};
Rectangle(5) = {1.55, 0.18, 0, 0.16, 0.20};
Rectangle(6) = {1.15, 1.02, 0, 0.18, 0.18};

air[] = BooleanDifference{ Surface{1}; Delete; }{ Surface{2, 3, 4, 5, 6}; };

Physical Surface(5) = {air[]};
Physical Surface(1) = {2, 3, 4};
Physical Surface(6) = {5};
Physical Surface(7) = {6};

Mesh.MeshSizeMin = lc;
Mesh.MeshSizeMax = lc;
