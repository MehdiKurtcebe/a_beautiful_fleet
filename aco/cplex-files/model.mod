string outputFile = ...;
string objFile = ...;

int zones = ...;			// num of zones
int beautificators = ...;	// num of beautificators
int t_max = ...;			// num of time slots

// sets
range Z = 1..zones;
range B = 1..beautificators;
range T = 0..t_max;

int z0[B] = ...;	// starting zone for each beautificator

int nOUT[Z] = ...;					// num of scooters out of the hotspot at t=0 for each zone
int nHOT[Z] = ...;					// num of scooters in the hotspot at t=0 for each zone
int n[z in Z] = nOUT[z] + nHOT[z];	// total num of scooters at t=0 for each zone

int mBEAU = ...;		// num of time slots required for beautification
int mHOT = ...;			// num of time slots required for bring-to-hotspot
int mMOVE[Z][Z] = ...;	// num of time slots required for moving from a zone to another zone
int mWAIT = ...;		// num of time slots required for waiting

// profits
float piBEAU = ...;			// profit of beautification
float piHOT = ...;			// profit of bring-to-hotspot
float piMOVE[Z][Z] = ...;	// profit of moving from a zone to another zone
float piWAIT = ...;			// profit of waiting

// node
tuple node {
  int z;	// zone
  int t;	// time slot
}

// arc
tuple arc {
  node nFrom;
  node nTo;
  float pi;
}

// set of nodes
{node} N = {<z, t> | z in Z, t in T};

// set of arcs
{arc} A_BEAU = { <<z, t>, <z, t + mBEAU>, piBEAU>
    | z in Z, t in T
    : <z, t> in N, <z, t + mBEAU> in N };
{arc} A_HOT = { <<z, t>, <z, t + mHOT>, piHOT>
    | z in Z, t in T
    : <z, t> in N, <z, t + mHOT> in N };
{arc} A_MOVE = { <<z1, t>, <z2, t + mMOVE[z1][z2]>, piMOVE[z1][z2]>
    | z1, z2 in Z, t in T
    : z1 != z2, <z1, t> in N, <z2, t + mMOVE[z1][z2]> in N };
{arc} A_WAIT = { <<z, t>, <z, t + mWAIT>, piWAIT>
    | z in Z, t in T
    : <z, t> in N, <z, t + mWAIT> in N };
{arc} A = A_BEAU union A_HOT union A_MOVE union A_WAIT;

// forward stars and backward stars
{arc} FW[z in Z][t in T] = {a | a in A: a.nFrom == <z, t>};
{arc} BW[z in Z][t in T] = {a | a in A: a.nTo == <z, t>};

// decision variables
dvar boolean x[B][A];

// Objective function
maximize sum(b in B, a in A) (
    a.pi * x[b][a]
);

// constraints
subject to {
  // each beautificator starts in his z0 at t=0
  forall(z in Z, b in B) {
    sum(a in FW[z][0]) x[b][a] == (z == z0[b] ? 1 : 0);
  }
  
  // flow conservation
  forall(z in Z, t in T, b in B: 0 < t < t_max) {
    sum(a in BW[z][t]) x[b][a] == sum(a in FW[z][t]) x[b][a];
  }
  
  // upper bound on the num of bring-to-hotspot actions
  forall(a in A_HOT) {
    sum(b in B) x[b][a]
    <=
    nOUT[a.nFrom.z]
    -
    sum(b in B, a2 in A_HOT: a2.nFrom.z == a.nFrom.z && a2.nFrom.t < a.nFrom.t)
            x[b][a2];
  }
  
  // upper bound on the num of beutification actions
  forall(a in A_BEAU) {
    sum(b in B) x[b][a]
    <=
    n[a.nFrom.z]
    -
    sum(b in B, a2 in A_HOT: a2.nFrom.z == a.nFrom.z && a2.nFrom.t < a.nFrom.t)
      x[b][a2]
    -
    sum(b in B, a2 in A_BEAU: a2.nFrom.z == a.nFrom.z && a2.nFrom.t < a.nFrom.t)
      x[b][a2];
  }
}

execute { 
  var scooters = 0; 
  for (var z in Z) { 
    scooters += n[z]; 
  } 
  
  var objf = new IloOplOutputFile("./results/" + objFile); 
  objf.writeln("-----CPLEX-----"); 
  objf.writeln("Total Beautificators: " + beautificators); 
  for (var z in z0) { 
    objf.writeln("Beautificator " + z + " Start Zone: " + z0[z]); 
  } 
  objf.writeln("Total Scooters: " + scooters); 
  objf.writeln("Total Profit: " + cplex.getObjValue()); 
  objf.close(); 

  // her guzellestirici icin ayri dosya olustur 
  for (var b in B) { 
    // Dosya adini olustur 
    var filename = "./results/" + outputFile + "_" + b + ".csv"; 

    // Dosyayi ac 
    var f = new IloOplOutputFile(filename); 

    // Header satirini yaz 
    f.writeln("Beautificator,Zone_From,Time_From,Zone_To,Time_To,Action"); 

    // Zaman dilimlerine gore hareketleri yazdir
    for (var t = 0; t <= t_max; t++) {
      for (var a in A) {
        if (x[b][a] > 0 && a.nFrom.t == t) { 
          f.write(b + "," + a.nFrom.z + "," + a.nFrom.t + "," + a.nTo.z + "," + a.nTo.t + ","); 
          if (A_BEAU.contains(a)) { 
            f.writeln("BEAU"); 
          } else if (A_HOT.contains(a)) { 
            f.writeln("HOT"); 
          } else if (A_MOVE.contains(a)) { 
            f.writeln("MOVE"); 
          } else if (A_WAIT.contains(a)) { 
            f.writeln("WAIT"); 
          } else { 
            f.writeln(""); 
          } 
        } 
      }
    }

    // Dosyayi kapat 
    f.close(); 
  } 
}