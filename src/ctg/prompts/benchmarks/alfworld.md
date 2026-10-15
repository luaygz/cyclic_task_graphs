ALFWorld is a text-based household simulator built on TextWorld from ALFRED scenes. Each episode places the agent in the middle of one room (kitchen, living room, bedroom or bathroom), lists the furniture and receptacles in it, and states one household task. The episode is won the moment the task's goal condition holds; there is no "done" or "submit" command.

**Task families.** Every task is one of six kinds:
- place an object: "put some book on sidetable", "put a cellphone in armchair";
- clean, then place: "put a clean butterknife in countertop" (clean it with a sink basin first);
- heat, then place: "heat some apple and put it in diningtable", "put a hot mug in sidetable" (heat it with a microwave first);
- cool, then place: "cool some apple and put it in countertop", "put a cool cup in sidetable" (cool it with a fridge first);
- examine under a light: "look at alarmclock under the desklamp", "examine the keychain with the desklamp" (hold the object, then turn on the lamp where it stands);
- place two objects: "put two alarmclock in dresser", "find two glassbottle and put them in shelf" (one object at a time).

"some apple" or "a mug" means any object of that type; the goal is satisfied by whichever instance ends up in the target receptacle.

**Partial observability.** The agent sees what is on or in a receptacle only while standing at it. Objects must be found by going to receptacles: arriving at an open or lidless receptacle lists its contents, while a closed one (drawer, cabinet, fridge, microwave, safe) reports "The drawer 1 is closed." and must be opened first. The object a task needs may be in any receptacle, so searching usually takes several moves. An object can only be taken from, or put into, an openable receptacle while it is open.

**Names.** Every receptacle and object has an exact numbered name, such as `countertop 2`, `cabinet 13`, `mug 1` or `desklamp 1`. Commands must use these names exactly as they appear in observations. Instances of the same type are distinct objects (`apple 1` is not `apple 2`), and the target receptacle type in a task may have several instances (`shelf 1` ... `shelf 6`), any of which satisfies the goal.

**Inventory.** The agent holds at most one object at a time. It must put the held object down before taking another. Heating, cooling and cleaning are applied to the held object while standing at the appliance.

**Commands.** Only commands admissible in the current state can be executed; each tool lists them.
- `look`: describe the current location.
- `inventory`: show the held object.
- `go to <receptacle>`: walk to a receptacle and see what is on or in it.
- `open <receptacle>` and `close <receptacle>`.
- `examine <receptacle>` or `examine <object>`: describe a receptacle, the held object, or an object in view.
- `take <object> from <receptacle>`: pick up an object (hands must be empty).
- `move <object> to <receptacle>`: put the held object in or on the receptacle at the current location.
- `heat <object> with <microwave>`, `cool <object> with <fridge>`, `clean <object> with <sinkbasin>`: process the held object at that appliance.
- `use <object>`: switch a device such as a desk lamp or floor lamp on or off.
- `slice <object> with <knife>`: slice an object at the current location while holding a knife.

**State changes are not confirmed.** The game narrates heating, cooling and cleaning ("You heat the mug 1 using the microwave 1.") but never confirms the object's new state: the inventory shows only the object's name, and examining it is unreliable (a heated cup may still read "This is a normal cup 1."). Track which objects have been processed; one successful heat, cool or clean command is enough. The processing must happen before the object is placed in the target receptacle.

**Budget.** Every tool call counts toward the episode's tool-call budget, including a command that is rejected because it is not admissible (a rejected command does not change the game state).
