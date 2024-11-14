import math
from typing import Optional, Union
from pyDOE import lhs
import numpy as np
from numpy import sin,cos,pi

import gym
from gym import logger, spaces
# from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
import matplotlib.pyplot as plt
import datetime
import casadi as ca
from casadi import *



class ParticleMass(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None, method = None):
        super(ParticleMass, self).__init__()

        self.dt = 0.01
        self.x_threshold = 8.0
        self.episode_steps = 0

        self.R0 = 1.5
        self.vel = 2.5
        self.k = 1
        self.lamda = 2

        self.d_goal = 0.5
        self.d_min = 0.75    #0.75 FOR NEXT TIME

        # Vehicle state
        self.x = 0.0
        self.y = 0.0
        self.u = 0.0
        self.v = 0.0

        # Goal state
        self.xgoal = 0.0
        self.ygoal = 0.0

        # Obstacle state
        self.xobs = 0.0
        self.yobs = 0.0

        # relative distances
        self.dx = 0.0
        self.dy = 0.0
        self.dx_obs = 0.0
        self.dy_obs = 0.0

        self.state = np.array([self.u,self.v,self.dx,self.dy,self.dx_obs,self.dy_obs]).astype(np.float32)
        
        high_action = np.array([2.0, 
                                2.0],
                                dtype=np.float32,
                                )


        high = np.array(
        [
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            self.x_threshold * 2,
            self.x_threshold * 2,
            self.x_threshold * 2,
            self.x_threshold * 2

        ],dtype=np.float32)
        
        # Define action and observation space
        self.action_space = spaces.Box(-high_action, +high_action, (2,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.render_mode = render_mode

        self.screen_width = 500
        self.screen_height = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.method = method

        self.steps_beyond_terminated = None

    def step(self, action):

        assert self.state is not None, "Call reset before using step method."

        reason = 'nothing'
        acc_rl = np.clip(action, -4.0, +4.0)

        if self.method == 'RL-CBF':
            acc = self.optimal_control_casadi(acc_rl)
        else:
            acc = acc_rl

        u,v,dx,dy,dxo,dyo = self.state
        x = self.xgoal - dx 
        y = self.ygoal - dy

        # dynamics
        x += self.dt * u
        y += self.dt * v
        u += self.dt * acc[0]
        v += self.dt * acc[1]

        dx = self.xgoal - x
        dy = self.ygoal - y
        dxo = self.xobs - x
        dyo = self.yobs - y

        heading = arctan2(v,u)
        goal_heading = arctan2(dy,dx)

        d_the = (goal_heading-heading)**2
        d_goal = np.sqrt((self.xgoal-x)**2 + (self.ygoal-y)**2)
        d_obs = np.sqrt((self.xobs-x)**2 + (self.yobs-y)**2)
        
        bound_terminate = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or y < -self.x_threshold
            or y > self.x_threshold
        )

        goal_terminate = bool(d_goal < self.d_goal)
        collision_terminate = bool(d_obs < self.d_min)

        truncate = bool(self.episode_steps >= 1000)
        terminate = bound_terminate or collision_terminate
        
        done = terminate or truncate or goal_terminate

        # reward
        if not done:
            reward = -12*(d_the) - 0.6*(d_goal)**2 - 1
            self.episode_steps+=1
            reason = 'working fine'
        elif self.steps_beyond_terminated is None:
            reward = 0.0
            if bound_terminate :
                reward = -10000
                reason = 'out of bounds'
            elif collision_terminate:
                reward = -10000
                reason = 'collision'
            elif goal_terminate:
                reward = 20000
                reason = 'goal reached'
            elif truncate and not goal_terminate:
                reward = -5000
                reason = 'max steps'
         
            self.steps_beyond_terminated = 0
        else:
            self.steps_beyond_terminated += 1


        self.state = np.array([u,v,dx,dy,dxo,dyo],dtype=np.float32)

        obs = self.state

        return obs, reward, done, self.episode_steps, reason

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        
        # Agent
        self.x = 0 + np.random.uniform(-1,1,1)[0]
        self.y = 0 + np.random.uniform(-1,1,1)[0] 
        self.u = 0.0
        self.v = 0.0
        
        # Goal
        self.xgoal = 6 + np.random.uniform(-1,1,1)[0] 
        self.ygoal = 6 + np.random.uniform(-1,1,1)[0] 

        # Obstacle
        self.xobs = 3 
        self.yobs = 3 

        self.dx = self.xgoal-self.x 
        self.dy = self.ygoal-self.y
        self.dx_obs = self.xobs-self.x
        self.dy_obs = self.yobs-self.y

        self.state = np.array([self.u,self.v,self.dx,self.dy,self.dx_obs,self.dy_obs]).astype(np.float32)

        self.steps_beyond_terminated = None
        self.episode_steps = 0

        return self.state

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()
  

        world_width = self.x_threshold * 2
        world_height = self.x_threshold * 2
        scalex = self.screen_width / world_width
        scaley = self.screen_height / world_height

        cartwidth = 20.0
        cartheight = 20.0

        if self.state is None:
            return None

        x = self.state
        # xc = self.xgoal - x[0]
        # yc = self.ygoal - x[1]
        xc = self.xgoal - x[2]
        yc = self.ygoal - x[3]
        # rxobs = np.array([x[4],x[5]], dtype=np.float32)
        # ryobs = np.array([x[6],x[7]], dtype=np.float32)

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

        cartx = xc * scalex + self.screen_width / 2.0  # MIDDLE OF CART
        carty = yc * scaley + self.screen_height / 2.0  # TOP OF CART

        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))
        
        goalx = self.xgoal * scalex + self.screen_width / 2.0
        goaly = self.ygoal * scaley + self.screen_height / 2.0
        gfxdraw.filled_circle(
            self.surf,
            int(goalx),
            int(goaly),
            5,
            (255, 0, 0),
        )


        obsx1 =  (self.xobs) * scalex + self.screen_width / 2.0 
        obsy1 =  (self.yobs) * scaley + self.screen_height / 2.0 
        gfxdraw.filled_circle(
            self.surf,
            int(obsx1),
            int(obsy1),
            15,
            (0, 0, 255),
        )

        self.font = pygame.font.SysFont("Arial", 20)
        # self.text = self.font.render("Episode: {}    Reward: {:0.1f}    Steps: {} ".format(ep,rw,st), True, (0,0,0)) # {:0.2f}   {:0.2f}   {:0.2f}
        # self.text2 = self.font.render("{:0.2f}   {:0.2f} \t\t.... {:0.2f}   {:0.2f}   {:0.2f}".format((xc),(xobs),(rxobs),(ryobs),self.c), True, (0,0,0))  

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        # self.screen.blit(self.text, (3, 3))
        # self.screen.blit(self.text2, (40, 40))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.update()
            pygame.display.flip()
            

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def plot_trajectory(self):
        
        x = self.xgoal - self.state[2]
        y = self.ygoal - self.state[3]

        return x,y,self.xgoal,self.ygoal,self.xobs,self.yobs

    def optimal_control_casadi(self,u_rl):
        
        x = self.xgoal - self.state[2]
        y = self.ygoal - self.state[3]
        u = self.state[0]
        v = self.state[1]

        u_rl = np.array(u_rl).reshape((2,1))

        # Safety barrier 
        h = (x-self.xobs)**2 + (y-self.yobs)**2 - self.R0**2
        h_dot = 2*(x-self.xobs)*u + 2*(y-self.yobs)*v

        # Lie Derivatives
        L2fh = 2*(u**2) + 2*(v**2)
        LgLfh = np.array([[2*u , 2*v]], dtype=np.float32)
        
        opti = ca.Opti()
        u = opti.variable(2)

        opti.minimize(0.5*(dot((u-u_rl),(u-u_rl))))

        # Constraint
        opti.subject_to(dot(LgLfh.T,u) + L2fh + self.lamda*(h_dot + self.k*h)>= 0)

        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver('ipopt',option)

        sol = opti.solve()
        u_opt = sol.value(u)

        return u_opt
    
    def reset_random(self, seed: Optional[int] = None,):
        super().reset(seed=seed)
        
        # Agent
        lb1 = np.array([-6,-6])
        ub1 = np.array([-4,6])

        lb2 = np.array([4,-6])
        ub2 = np.array([6,6])

        start = lb1 + (ub1-lb1)*lhs(2,1)
        goal = lb2 + (ub2-lb2)*lhs(2,1)


        self.x = start[0,0]
        self.y = start[0,1]
        self.u = 0.0
        self.v = 0.0
        
        # Goal
        self.xgoal = goal[0,0] 
        self.ygoal = goal[0,1] 

        # Obstacle
        self.xobs = 0 
        self.yobs = np.random.uniform(-4,4,1)[0] 

        self.dx = self.xgoal-self.x 
        self.dy = self.ygoal-self.y
        self.dx_obs = self.xobs-self.x
        self.dy_obs = self.yobs-self.y

        self.state = np.array([self.u,self.v,self.dx,self.dy,self.dx_obs,self.dy_obs]).astype(np.float32)

        self.steps_beyond_terminated = None
        self.episode_steps = 0

        return self.state