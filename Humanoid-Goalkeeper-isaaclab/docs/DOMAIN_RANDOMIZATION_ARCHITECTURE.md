# Domain Randomization Architecture: Direct Workflow vs Manager-Based

**Answer:** The Isaac Lab implementation uses **DIRECT WORKFLOW** (inline, procedural).

---

## What This Means

### Direct Workflow (Current Implementation ✅)
Randomization calls are **directly embedded** in the environment methods:
- No separate manager class
- Called inline where needed
- Procedural, straightforward
- Easy to trace and debug

### Manager-Based (Alternative)
Would use a separate `DomainRandomizationManager` class:
- Separate module to handle all randomization
- Called from environment
- More abstraction, but more complex

---

## Current Architecture (Direct Workflow)

```
GoalkeeperEnv (extends DirectRLEnv)
│
├─ _pre_physics_step()
│  └─ Delay setup [DIRECT CALL]
│
├─ _apply_action()
│  └─ Joint injection randomization [DIRECT CALL]
│
├─ _reset_idx()
│  ├─ Kp factor randomization [DIRECT CALL]
│  ├─ Kd factor randomization [DIRECT CALL]
│  └─ Actuation offset randomization [DIRECT CALL]
│
├─ _get_observations()
│  └─ Ball randomization [DIRECT CALL via _randomize_balls()]
│
└─ __init__() / _setup_buffers()
   ├─ Payload mass randomization [DIRECT CALL]
   ├─ Link mass randomization [DIRECT CALL]
   ├─ CoM displacement randomization [DIRECT CALL]
   └─ Initial joint pos randomization [DIRECT CALL in _reset_dofs()]
```

---

## Code Flow: Where Each Randomization Happens

### 1. INITIALIZATION (One-time at env creation)

**File:** `goalkeeper_env.py` lines 500-530

```python
def _setup_buffers(self, num_envs: int):
    """Initialize randomization buffers."""
    # Kp/Kd factors
    self.Kp_factors = torch.ones(...)
    if self.cfg.randomize_kp:
        self.Kp_factors = torch_rand_float(...)  # [DIRECT]
    
    self.Kd_factors = torch.ones(...)
    if self.cfg.randomize_kd:
        self.Kd_factors = torch_rand_float(...)  # [DIRECT]
    
    # Actuation offset buffer
    self.actuation_offset = torch.zeros(...)
```

---

### 2. RESET (When episode ends, env_ids reset)

**File:** `goalkeeper_env.py` lines 187-218

```python
def _reset_idx(self, env_ids: torch.Tensor):
    """Called when episodes terminate."""
    
    # Reset physical state
    self._reset_dofs(env_ids)          # Resets with joint pos randomization
    self._reset_root_states(env_ids)
    
    # Randomize control parameters
    if self.cfg.randomize_kp:
        self.Kp_factors[env_ids] = torch_rand_float(...)  # [DIRECT]
    
    if self.cfg.randomize_kd:
        self.Kd_factors[env_ids] = torch_rand_float(...)  # [DIRECT]
    
    if self.cfg.randomize_actuation_offset:
        self.actuation_offset[env_ids] = torch_rand_float(...)  # [DIRECT]
```

**What gets randomized per reset:**
- ✅ Kp gains
- ✅ Kd gains  
- ✅ Actuation offset
- ✅ Initial joint positions (in _reset_dofs)

**What gets randomized once at init (then scaled per reset):**
- ✅ Payload mass (applied via mass_props)
- ✅ Link mass factors (applied via mass_props)
- ✅ CoM displacement (applied via mass_props)

---

### 3. STEP (Every simulation step)

**File:** `goalkeeper_env.py` lines 95-127

```python
def _pre_physics_step(self, actions):
    """Called before each physics step."""
    # Delay simulation
    if self.cfg.delay:
        delay_steps = torch.randint(0, decimation, ...)  # [DIRECT]
        self.delayed_actions[i] = interpolate(old, new, delay)
    
def _apply_action(self):
    """Called each physics step."""
    action = self.delayed_actions[step]
    
    # Joint injection (noise on target angles)
    if self.cfg.randomize_joint_injection:
        self.joint_injection = torch_rand_float(...)  # [DIRECT]
    
    torques = self._compute_torques(action)
    self._robot.set_joint_effort_target(torques)
```

**What gets randomized per step:**
- ✅ Joint injection (angle noise)
- ✅ Control delay (in delayed_actions)

---

### 4. OBSERVATION (Every step, when observations computed)

**File:** `goalkeeper_env.py` lines 262-330

```python
def _get_observations(self):
    """Called every step to build observation."""
    
    # Ball randomization (physics)
    self._randomize_balls()  # [DIRECT]
    
    # Ball visibility randomization (observation)
    flying = compute_visibility()
    random_vanish = (step > threshold)
    obs = cat(ball * flying * random_vanish, ...)
```

**What gets randomized per observation:**
- ✅ Ball friction (in _randomize_balls)
- ✅ Ball restitution (in _randomize_balls)
- ✅ Ball observation visibility mask

---

## Randomization Timing Summary

| Parameter | When Randomized | Frequency | Code Location |
|-----------|-----------------|-----------|---------------|
| **Joint Injection** | Every step | Every control step | `_apply_action()` line 116 |
| **Actuation Offset** | Each reset | When episode ends | `_reset_idx()` line 212 |
| **Delay** | Every step | Per control step | `_pre_physics_step()` line 102 |
| **Kp Factor** | Each reset | When episode ends | `_reset_idx()` line 204 |
| **Kd Factor** | Each reset | When episode ends | `_reset_idx()` line 208 |
| **Joint Initial Pos** | Each reset | When episode ends | `_reset_dofs()` implicit |
| **Payload Mass** | Init + per reset | Rare (via mass apply) | `_setup_buffers()` line 515 |
| **Link Mass** | Init + per reset | Rare (via mass apply) | `_setup_buffers()` line 520 |
| **CoM Displacement** | Init only | Once at start | `_setup_buffers()` line 525 |
| **Ball Friction** | Each reset/step | In `_randomize_balls()` | `_get_observations()` line 267 |
| **Ball Restitution** | Each reset/step | In `_randomize_balls()` | `_get_observations()` line 267 |

---

## Why Direct Workflow?

### Advantages ✅
1. **Straightforward**: Easy to trace where randomization happens
2. **Efficient**: No extra function calls or indirection
3. **Clear intent**: Randomization is explicit at call site
4. **Debuggable**: Set breakpoints directly on randomization code
5. **Flexible**: Different parameters randomized at different rates (step vs reset vs init)

### Disadvantages ❌
1. **Scattered**: Randomization logic spread across multiple methods
2. **Hard to modify**: Change all randomization → edit multiple methods
3. **No single interface**: No one place to enable/disable all randomization
4. **Repetitive**: Config checks (if cfg.randomize_X) scattered throughout

---

## Alternative: Manager-Based Architecture (Not Used)

If implemented, would look like:

```python
class DomainRandomizationManager:
    """Centralized domain randomization handler."""
    
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        # Allocate buffers
        self.Kp_factors = torch.ones(...)
        self.Kd_factors = torch.ones(...)
        ...
    
    def randomize_on_reset(self, env_ids):
        """Call at reset."""
        if self.cfg.randomize_kp:
            self.Kp_factors[env_ids] = ...
        if self.cfg.randomize_kd:
            self.Kd_factors[env_ids] = ...
        ...
    
    def randomize_on_step(self):
        """Call each step."""
        if self.cfg.randomize_joint_injection:
            self.joint_injection = ...
        ...
    
    def randomize_ball(self):
        """Call when computing obs."""
        ...

# In environment:
class GoalkeeperEnv(DirectRLEnv):
    def __init__(self, ...):
        self.domain_rand = DomainRandomizationManager(...)
    
    def _reset_idx(self, env_ids):
        self.domain_rand.randomize_on_reset(env_ids)  # Single call
    
    def _apply_action(self):
        self.domain_rand.randomize_on_step()  # Single call
```

**Pros:** Cleaner, centralized control  
**Cons:** Extra abstraction layer, more code

---

## Current Implementation Assessment

The **direct workflow** is appropriate here because:

1. ✅ Different parameters randomized at different frequencies (step/reset/init)
   - Manager would need complex scheduling
   - Direct is simpler

2. ✅ Tight integration with physics and control loops
   - Randomization must happen at exact point in control pipeline
   - Direct allows precise placement

3. ✅ Performance-critical (1000s of envs)
   - Direct has minimal overhead
   - Manager adds extra function calls

4. ✅ Small, focused environment (1 task)
   - Not a complex multi-task framework
   - Manager would be overkill

---

## Comparison to Original (Isaac Gym)

**Original (Isaac Gym):** Also uses **direct workflow**
- Same structure: randomization scattered through methods
- Same timing: reset → apply → step → obs
- Same philosophy: simplicity over abstraction

---

## Summary

**Architecture:** ✅ **Direct Workflow (Inline)**

```
No Manager Class
    ↓
Randomization calls directly in:
  - _pre_physics_step()  (Delay)
  - _apply_action()      (Joint injection)
  - _reset_idx()         (Kp, Kd, offset)
  - _get_observations()  (Ball)
  - __init__()           (Init-time params)
```

This is pragmatic, efficient, and matches the original Isaac Gym implementation.
