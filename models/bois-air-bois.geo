SetFactory("OpenCASCADE");

lc = 0.05;

// Domaine 2D : bois | air | bois, avec trois ponts de bois dans l'air.
Rectangle(1) = {0, 0, 0, 1, 1}; // bois gauche
Rectangle(2) = {1, 0, 0, 1, 1}; // air central
Rectangle(3) = {2, 0, 0, 1, 1}; // bois droit

// Ponts de bois dans la couche d'air, espaces horizontalement.
Rectangle(4) = {1.10, 0.40, 0, 0.14, 0.20};
Rectangle(5) = {1.32, 0.40, 0, 0.14, 0.20};
Rectangle(6) = {1.54, 0.40, 0, 0.14, 0.20};
Rectangle(7) = {1.76, 0.40, 0, 0.14, 0.20};

air[] = BooleanDifference{ Surface{2}; Delete; }{ Surface{4, 5, 6, 7}; };

Coherence;

Physical Surface("bois", 1) = {1, 3, 4, 5, 6, 7};
Physical Surface("air", 5) = {air[]};

Mesh.MeshSizeMin = lc;
Mesh.MeshSizeMax = lc;
Mesh.Algorithm = 6;
