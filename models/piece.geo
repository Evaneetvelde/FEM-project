SetFactory("OpenCASCADE");
lc = 0.03;

// --- Dimensions ---
L = 1.0; // Largeur totale
H = 1.0; // Hauteur totale
th = 0.04; // Épaisseur des murs

// 1. Dalle de sol principale
Rectangle(1) = {0, 0, 0, L, H};

// 2. Murs internes (selon le schéma)
// Mur horizontal central (avec une porte à droite)
Rectangle(2) = {0, 0.5 - th/2, 0, 0.7, th}; 

// Mur vertical (avec une porte au milieu)
Rectangle(3) = {0.5 - th/2, 0, 0, th, 0.4}; // Partie basse
Rectangle(4) = {0.5 - th/2, 0.6, 0, th, 0.4}; // Partie haute

// 3. Définition des surfaces
// On soustrait les murs du sol pour que le maillage ne se chevauche pas
floor[] = BooleanDifference{ Surface{1}; Delete; }{ Surface{2, 3, 4}; };

// 4. Groupes Physiques
// ID 1 = Bois (Sol), ID 2 = Béton (Murs)
Physical Surface(1) = {floor[]};   
Physical Surface(2) = {2, 3, 4};   

Mesh.MeshSizeMin = lc;
Mesh.MeshSizeMax = lc;