"""
This is a pebble game algorithm written in python, translated from the original program in Fortran.
To run this program, you need to have python and cython installed.
By Leyou Zhang, 2016, 04
"""

class lattice(object):
    def __init__(self):
        self.digraph = {} # directed graph
        self.graph = {} # undirected graph (complete)
        self.cluster={} # cluster information
        self.bond = []
        self.stress = []
        self.visited = []
        self.statistics = {}

    def clear(self):
        self.__init__()

    def add_bond(self, x, y):

        """
        :param x: a site
        :param y: a different site
        :return: if the new bond is independent: True, otherwise False.
        """

        # no self-loop bonds:
        if x == y:
            raise ValueError('add_bond must have two different sites')

        # skip pebble game if this edge already exists
        # we will be adding bonds from a dict of sets where some bonds are represented twice
        if x in self.graph:
            if y in self.graph[x]:
                return True

        # smaller site first:
        x,y = sorted((x,y))
        # update bonds:
        self.bond.append((x,y))

        # update complete graph
        sites = self.graph.keys()  # potentially remove this
        if x not in sites:
            self.graph[x] = [y]
        else:
            self.graph[x].append(y)
        if y not in sites:
            self.graph[y] = [x]
        else:
            self.graph[y].append(x)
            
        # update directed graph
        sites = self.digraph.keys()
        if x not in sites:
            self.digraph[x] = [[y],2]
            if y not in sites:
                self.digraph[y] = [[],3]
            return True
        elif y not in sites:
            self.digraph[y] = [[x],2]
            return True
        elif self.collect_six_pebble(x,y):
            if y not in self.digraph[x][0]:
                self.digraph[x][0].append(y)
            try:
                self.digraph[x][1] = self.digraph[x][1] -1
            except Exception as f:
                raise KeyError(f.message)
            return True
        else:
            return False

            # check independence

    def depth_first_search(self, x, y, z = False, status = 'start'):

        if status == 'start':
            if not z:
                self.visited = [x,y]
            else:
                self.visited = [x,y,z]
        else:
            self.visited.append(x)

        # exclude y (or y,z) in the search
        if not z:
            if x == y:
                raise ValueError('depth_first_search must have two or three different sites')
            for i in self.digraph[x][0]:
                if i not in self.visited:
                    if self.digraph[i][1] > 0:
                        tree = [i]
                        return tree
                    tree = self.depth_first_search(i,y,status='next')
                    if tree:
                        return [i]+tree
        else:
            if x == y or x == z or y == z:
                raise ValueError('depth_first_search must have two or three different sites')
            for i in self.digraph[x][0]:
                if i not in self.visited:
                    if self.digraph[i][1] > 0:
                        tree = [i]
                        return tree
                    tree = self.depth_first_search(i,y, z = z,status='next')
                    if tree:
                        return [i]+tree
        return None


    def collect_one_pebble(self, x, y):

        """
        :param x: a site
        :param y: a different site
        :return: if the one pebble can be collected, return True, otherwise False.
        """

        sites = self.graph.keys()
        if x in sites:
            tree = self.depth_first_search(x,y)
            if tree:
                self.digraph[x][1] += 1
                while tree:
                    site = tree.pop(0)
                    self.digraph[x][0].remove(site)
                    self.digraph[site][0].append(x)
                    x = site
                self.digraph[site][1] += - 1
                return True
            else:
                return False
        else:
            raise ValueError('site %d is not in the lattice.'%x)

    def collect_six_pebble(self, x, y):

        """
        :param x: a site
        :param y: a different site
        :return: if the six pebble can be collected, return True, otherwise False.
        """

        if x == y:
            raise ValueError('collect_six_pebble must have two different sites')

        freex = self.digraph[x][1]
        freey = self.digraph[y][1]
        while freex < 3:
            if self.collect_one_pebble(x,y):
                freex += 1
            else:
                break
        while freey < 3:
            if self.collect_one_pebble(y,x):
                freey += 1
            else:
                break

        # go through each of the first neighbors and accumulate a pebble
        num_neighbors = len(self.digraph[x][0])
        num_neighbors_collected = 0
        for i in self.digraph[x][0]:
            if self.collect_one_pebble(i, x):
                num_neighbors_collected += 1

        num_neighbors2 = 0
        num_neighbors_collected2 = 0
        for i in self.digraph[y][0]:
            if i not in self.digraph[x][0]:
                num_neighbors2 += 1
                if self.collect_one_pebble(i, y):
                    num_neighbors_collected2 += 1
                
        if freex==3 and freey==3 and num_neighbors_collected==num_neighbors and num_neighbors_collected2==num_neighbors2:
            return True
        else:
            return False

if __name__ == "__main__":
    l = lattice()
    print(l.add_bond(1,2))
    print(l.add_bond(2,4))
    print(l.add_bond(3,4))
    print(l.add_bond(1,3))
    print(l.add_bond(1,4))
    print(l.add_bond(2,3))
    print(l.add_bond(1,2))


